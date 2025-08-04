from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
)
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import nltk
from nltk.corpus import stopwords
from collections import Counter
import lxml.html
import commonmark
import ast
import re
import requests
import os
import pathlib
import yaml
import json
import markdown
from bs4 import BeautifulSoup
import docutils.core
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
from typing import List, Dict, Optional, Any


class DocumentationIngestor:
    def format_code(self, code: str) -> str:
        """
        Formats Python code using black for style compliance.
        """
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp.flush()
            subprocess.run(["black", tmp.name], capture_output=True)
            tmp.seek(0)
            formatted = tmp.read()
        return formatted

    def add_error_handling(self, code: str) -> str:
        """
        Adds basic try/except error handling to function bodies (Python).
        """
        import ast
        import astor

        class ErrorHandler(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if not any(isinstance(stmt, ast.Try) for stmt in node.body):
                    try_block = ast.Try(
                        body=node.body,
                        handlers=[
                            ast.ExceptHandler(type=None, name=None, body=[ast.Pass()])
                        ],
                        orelse=[],
                        finalbody=[],
                    )
                    node.body = [try_block]
                return node

        tree = ast.parse(code)
        tree = ErrorHandler().visit(tree)
        return astor.to_source(tree)

    def check_security(self, code: str) -> bool:
        """
        Checks for basic security issues (e.g., eval, exec, unsafe input).
        """
        import re

        patterns = [r"eval\(", r"exec\(", r"os\.system", r"subprocess\.Popen"]
        for pat in patterns:
            if re.search(pat, code):
                return False
        return True

    def is_idempotent(self, code: str, snippet: str) -> bool:
        """
        Checks if snippet already exists in code to avoid duplication.
        """
        return snippet.strip() in code

    def context_aware_insert(
        self, code: str, snippet: str, location: str = "end", anchor: str = None
    ) -> str:
        """
        Inserts code snippet only if not present, and at contextually correct location.
        """
        if self.is_idempotent(code, snippet):
            return code
        return self.insert_code_snippet(code, snippet, location, anchor)

    def modify_code_ast(self, code: str, modification_fn) -> str:
        """
        Parses code into AST, applies modification_fn, and regenerates code.
        modification_fn should accept and return an AST tree.
        """
        import ast

        try:
            tree = ast.parse(code)
            tree = modification_fn(tree)
            import astor

            return astor.to_source(tree)
        except Exception as e:
            print(f"AST modification failed: {e}")
            return code

    def regex_replace_code(self, code: str, pattern: str, replacement: str) -> str:
        """
        Uses regex to find and replace code patterns.
        """
        import re

        return re.sub(pattern, replacement, code)

    def insert_code_snippet(
        self, code: str, snippet: str, location: str = "end", anchor: str = None
    ) -> str:
        """
        Inserts code snippet at specified location: 'start', 'end', or after anchor (function/class/module).
        """
        if location == "start":
            return snippet + "\n" + code
        elif location == "end":
            return code + "\n" + snippet
        elif anchor:
            import re

            # Insert after anchor (function/class/module name)
            pattern = rf"(def {anchor}\b.*?:|class {anchor}\b.*?:)"
            match = re.search(pattern, code)
            if match:
                idx = match.end()
                return code[:idx] + "\n" + snippet + code[idx:]
        return code

    def generate_code_template(self, template: str, context: dict) -> str:
        """
        Generates code from a template with placeholders replaced by context values.
        """
        return template.format(**context)

    def synthesize_code(self, description: str, model_name: str = "gpt2") -> str:
        """
        Generates code from a high-level description using a language model.
        """
        from transformers import pipeline

        generator = pipeline("text-generation", model=model_name)
        result = generator(description, max_length=128, do_sample=True)
        return result[0]["generated_text"]

    def find_and_adapt_example(self, query: str, codebase: list, context: dict) -> str:
        """
        Finds similar code examples and adapts them to the new component using semantic search and template replacement.
        """
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        code_emb = model.encode(codebase, convert_to_tensor=True)
        query_emb = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, code_emb, top_k=1)
        if hits and hits[0]:
            example = codebase[hits[0][0]["corpus_id"]]
            # Simple adaptation: replace placeholders
            return example.format(**context)
        return ""

    def plm_ner(self, text: str, model_name: str = "dslim/bert-base-NER") -> list:
        # Named Entity Recognition using BERT/transformers
        nlp = pipeline("ner", model=model_name)
        return nlp(text)

    def plm_qa(
        self,
        question: str,
        context: str,
        model_name: str = "distilbert-base-cased-distilled-squad",
    ) -> dict:
        # Question Answering using transformers
        qa = pipeline("question-answering", model=model_name)
        return qa(question=question, context=context)

    def ir_tfidf(self, query: str, docs: list) -> list:
        # TF-IDF based IR
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([query])
        scores = (X * query_vec.T).toarray().flatten()
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked if score > 0]

    def ir_embeddings(
        self, query: str, docs: list, model_name: str = "all-MiniLM-L6-v2"
    ) -> list:
        # Embeddings-based IR
        model = SentenceTransformer(model_name)
        doc_emb = model.encode(docs, convert_to_tensor=True)
        query_emb = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, doc_emb, top_k=5)
        return [docs[hit["corpus_id"]] for hit in hits[0]]

    def build_knowledge_graph(self, entities: list, relationships: list) -> nx.Graph:
        # Build a knowledge graph from entities and relationships
        G = nx.Graph()
        for ent in entities:
            G.add_node(ent)
        for rel in relationships:
            if "caller" in rel and rel["caller"] and rel["callee"]:
                G.add_edge(rel["caller"], rel["callee"], params=rel.get("params", []))
            elif "callee" in rel and rel["callee"]:
                G.add_node(rel["callee"])
        return G

    def reason_with_knowledge_graph(
        self, G: nx.Graph, source: str, target: str
    ) -> list:
        # Find reasoning path between entities/components/APIs
        try:
            path = nx.shortest_path(G, source=source, target=target)
            return path
        except Exception as e:
            print(f"Knowledge graph reasoning failed: {e}")
            return []

    def analyze_component_context(
        self,
        metadata: dict,
        dependencies: list,
        api_signatures: list,
        doc_sections: list,
    ) -> dict:
        """
        Infers component purpose and usage from extracted knowledge.
        """
        context = {}
        # Purpose: Use metadata, doc_sections, and API signatures
        context["purpose"] = metadata.get("summary", "") or (
            doc_sections[0] if doc_sections else ""
        )
        context["usage"] = api_signatures if api_signatures else []
        context["dependencies"] = dependencies
        # Usage example: Find code blocks or API usage
        context["usage_examples"] = (
            metadata.get("code_examples", []) if "code_examples" in metadata else []
        )
        return context

    def recognize_intent(
        self, user_input: str, project_requirements: list, component_context: dict
    ) -> dict:
        """
        Determines user's goal and matches with component capabilities.
        """
        import re

        intent = {}
        # Simple keyword matching for demo
        for req in project_requirements:
            if re.search(req, user_input, re.IGNORECASE):
                intent["goal"] = req
                break
        # Match with component purpose
        if "purpose" in component_context and intent.get("goal"):
            if intent["goal"].lower() in component_context["purpose"].lower():
                intent["match"] = True
            else:
                intent["match"] = False
        return intent

    def reason_integration(
        self, component_context: dict, intent: dict, existing_code: str = ""
    ) -> dict:
        """
        Develops a plan for integration, compares approaches, detects conflicts, and suggests modifications.
        """
        reasoning = {}
        # Compare integration approaches (demo: direct import vs API call)
        if "usage" in component_context:
            if any(
                "api" in sig.get("name", "").lower()
                for sig in component_context["usage"]
            ):
                reasoning["approach"] = "API integration"
            else:
                reasoning["approach"] = "Direct import"
        # Detect conflicts (demo: dependency overlap)
        if existing_code:
            import re

            for dep in component_context.get("dependencies", []):
                if re.search(dep.split("=")[0], existing_code):
                    reasoning.setdefault("conflicts", []).append(dep)
        # Suggest modifications (demo: add import or API call)
        if reasoning.get("approach") == "Direct import":
            reasoning["suggestion"] = "Add import statement for the component."
        elif reasoning.get("approach") == "API integration":
            reasoning["suggestion"] = "Add API call and handle response."
        return reasoning

    def parse_requirements_txt(self, file_path: str) -> list:
        deps = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("["):
                        pkg = line.split("#")[0].strip()
                        if pkg:
                            deps.append(pkg)
        except Exception as e:
            print(f"Failed to parse requirements.txt: {e}")
        return deps

    def parse_package_json(self, file_path: str) -> list:
        import json

        deps = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for section in ["dependencies", "devDependencies", "peerDependencies"]:
                    if section in data:
                        deps.extend([f"{k}@{v}" for k, v in data[section].items()])
        except Exception as e:
            print(f"Failed to parse package.json: {e}")
        return deps

    def parse_pom_xml(self, file_path: str) -> list:
        from xml.etree import ElementTree as ET

        deps = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for dep in root.findall(".//dependency"):
                group = (
                    dep.find("groupId").text if dep.find("groupId") is not None else ""
                )
                artifact = (
                    dep.find("artifactId").text
                    if dep.find("artifactId") is not None
                    else ""
                )
                version = (
                    dep.find("version").text if dep.find("version") is not None else ""
                )
                deps.append(f"{group}:{artifact}:{version}")
        except Exception as e:
            print(f"Failed to parse pom.xml: {e}")
        return deps

    def extract_dependency_sections(self, doc_content: str) -> list:
        # Look for sections like 'Requirements', 'Dependencies', 'Prerequisites'
        import re

        sections = re.findall(
            r"(?:Requirements|Dependencies|Prerequisites)[\s\S]+?(?=\n\w|$)",
            doc_content,
            re.IGNORECASE,
        )
        return sections

    def infer_implicit_dependencies(self, code: str, language: str = "python") -> list:
        deps = []
        if language == "python":
            import re

            imports = re.findall(r"import\s+(\w+)", code)
            from_imports = re.findall(r"from\s+(\w+)", code)
            deps.extend(imports)
            deps.extend(from_imports)
        elif language == "js":
            import re

            requires = re.findall(r'require\(["\'](\w+)["\']\)', code)
            imports = re.findall(r'import\s+(?:\w+\s+from\s+)?["\'](\w+)["\']', code)
            deps.extend(requires)
            deps.extend(imports)
        elif language == "java":
            import re

            imports = re.findall(r"import\s+([\w\.]+);", code)
            deps.extend(imports)
        return list(set(deps))

    def identify_code_blocks(self, doc_content: str, format: str = "markdown") -> list:
        # Detect code blocks in markdown, rst, html
        code_blocks = []
        if format == "markdown":
            import re

            code_blocks = re.findall(r"```[\w\W]*?```", doc_content)
        elif format == "rst":
            code_blocks = re.findall(r"::\n([\s\S]+?)(?=\n\S|$)", doc_content)
        elif format == "html":
            soup = BeautifulSoup(doc_content, "html.parser")
            code_blocks = [c.get_text() for c in soup.find_all("code")]
        return code_blocks

    def extract_api_signatures(self, code: str, language: str = "python") -> list:
        # Extract function/method signatures
        signatures = []
        if language == "python":
            import ast

            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        params = [arg.arg for arg in node.args.args]
                        returns = ast.unparse(node.returns) if node.returns else None
                        signatures.append(
                            {"name": node.name, "params": params, "returns": returns}
                        )
            except Exception as e:
                print(f"API signature extraction failed: {e}")
        else:
            # Simple regex for JS, etc.
            import re

            matches = re.findall(r"function\s+(\w+)\s*\(([^)]*)\)", code)
            for name, params in matches:
                param_list = [p.strip() for p in params.split(",") if p.strip()]
                signatures.append({"name": name, "params": param_list, "returns": None})
        return signatures

    def extract_param_descriptions(self, docstring: str) -> list:
        # Extract parameter descriptions from Python docstrings (Google/Numpy style)
        import re

        param_descs = []
        # Google style: Args:
        args_block = re.search(r"Args:\s*([\s\S]+?)(?=\n\w|$)", docstring)
        if args_block:
            lines = args_block.group(1).splitlines()
            for line in lines:
                m = re.match(r"\s*(\w+)\s*\(([^)]+)\):\s*(.*)", line)
                if m:
                    param_descs.append(
                        {
                            "name": m.group(1),
                            "type": m.group(2),
                            "description": m.group(3),
                        }
                    )
        # Numpy style: Parameters
        params_block = re.search(r"Parameters\s*-*\n([\s\S]+?)(?=\n\w|$)", docstring)
        if params_block:
            lines = params_block.group(1).splitlines()
            for line in lines:
                m = re.match(r"\s*(\w+)\s*:\s*([^\n]+)", line)
                if m:
                    param_descs.append(
                        {"name": m.group(1), "type": m.group(2), "description": ""}
                    )
        return param_descs

    def extract_relationships(self, code: str, language: str = "python") -> list:
        # Simple dependency extraction: function calls, parameter usage
        relationships = []
        if language == "python":
            import ast

            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if hasattr(node.func, "id"):
                            relationships.append(
                                {
                                    "caller": None,
                                    "callee": node.func.id,
                                    "params": [ast.unparse(arg) for arg in node.args],
                                }
                            )
            except Exception as e:
                print(f"Relationship extraction failed: {e}")
        # For other languages, regex or static analysis can be added
        return relationships

    def extract_key_terms(self, text: str, top_n: int = 10) -> list:
        # Simple keyword extraction using TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
        X = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
        return list(terms)

    def extract_named_entities(self, text: str) -> list:
        # Use spaCy for NER
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def extract_code_entities(self, code: str) -> dict:
        # Extract function names, class names, parameter names, data types using AST
        import ast

        result = {"functions": [], "classes": [], "params": [], "datatypes": []}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    result["functions"].append(node.name)
                    for arg in node.args.args:
                        result["params"].append(arg.arg)
                elif isinstance(node, ast.ClassDef):
                    result["classes"].append(node.name)
                elif isinstance(node, ast.AnnAssign):
                    if hasattr(node, "annotation"):
                        result["datatypes"].append(ast.unparse(node.annotation))
        except Exception as e:
            print(f"AST code entity extraction failed: {e}")
        return result

    def topic_modeling(self, docs: list, n_topics: int = 3) -> dict:
        # Use LDA for topic modeling
        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform(docs)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        topics = {}
        for idx, topic in enumerate(lda.components_):
            top_features = [
                vectorizer.get_feature_names_out()[i]
                for i in topic.argsort()[-10:][::-1]
            ]
            topics[f"Topic {idx+1}"] = top_features
        return topics

    def parse_html_lxml(self, content: str) -> Dict[str, Any]:
        tree = lxml.html.fromstring(content)
        title = tree.findtext(".//title") or ""
        summary = ""
        p = tree.find(".//p")
        if p is not None:
            summary = p.text_content()
        return {"title": title, "summary": summary}

    def parse_markdown_commonmark(self, content: str) -> Dict[str, Any]:
        parser = commonmark.Parser()
        ast_node = parser.parse(content)
        renderer = commonmark.HtmlRenderer()
        html = renderer.render(ast_node)
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find(["h1", "h2", "h3"])
        summary = soup.p.get_text() if soup.p else ""
        return {
            "title": title.get_text() if title else "",
            "summary": summary,
            "html": html,
        }

    def extract_with_regex(self, pattern: str, content: str) -> list:
        return re.findall(pattern, content, re.DOTALL)

    def parse_python_code_ast(self, code: str) -> Dict[str, Any]:
        result = {"docstrings": [], "functions": [], "classes": []}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    doc = ast.get_docstring(node)
                    result["functions"].append({"name": node.name, "docstring": doc})
                elif isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node)
                    result["classes"].append({"name": node.name, "docstring": doc})
                elif isinstance(node, ast.Module):
                    doc = ast.get_docstring(node)
                    if doc:
                        result["docstrings"].append(doc)
        except Exception as e:
            print(f"AST parsing failed: {e}")
        return result

    def parse_markdown(self, content: str) -> Dict[str, Any]:
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find(["h1", "h2", "h3"])
        summary = soup.p.get_text() if soup.p else ""
        return {
            "title": title.get_text() if title else "",
            "summary": summary,
            "html": html,
        }

    def parse_rst(self, content: str) -> Dict[str, Any]:
        parts = docutils.core.publish_parts(content, writer_name="html")
        soup = BeautifulSoup(parts["html_body"], "html.parser")
        title = soup.find(["h1", "h2", "h3"])
        summary = soup.p.get_text() if soup.p else ""
        return {
            "title": title.get_text() if title else "",
            "summary": summary,
            "html": parts["html_body"],
        }

    def parse_html(self, content: str) -> Dict[str, Any]:
        soup = BeautifulSoup(content, "html.parser")
        title = soup.title.get_text() if soup.title else ""
        summary = soup.p.get_text() if soup.p else ""
        return {"title": title, "summary": summary}

    def parse_plain_text(self, content: str) -> Dict[str, Any]:
        lines = content.splitlines()
        title = lines[0] if lines else ""
        summary = ""
        for line in lines[1:]:
            if line.strip():
                summary = line.strip()
                break
        return {"title": title, "summary": summary}

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception:
            # Fallback to PyPDF2
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ""
            except Exception as e:
                print(f"PDF parsing failed: {e}")
        # OCR fallback (if no text extracted)
        if not text:
            try:
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
            except Exception as e:
                print(f"OCR failed: {e}")
        return self.parse_plain_text(text)

    def parse_json_yaml(self, content: str) -> Dict[str, Any]:
        try:
            data = yaml.safe_load(content)
        except Exception:
            try:
                data = json.loads(content)
            except Exception as e:
                print(f"JSON/YAML parsing failed: {e}")
                return {}
        title = data.get("title", "") if isinstance(data, dict) else ""
        summary = data.get("description", "") if isinstance(data, dict) else ""
        return {"title": title, "summary": summary, "data": data}

    def parse_code_comments(
        self, content: str, language: str = "python"
    ) -> Dict[str, Any]:
        import re

        if language == "python":
            docstrings = re.findall(r'"""(.*?)"""', content, re.DOTALL)
            title = docstrings[0].splitlines()[0] if docstrings else ""
            summary = docstrings[0] if docstrings else ""
        elif language == "js":
            jsdoc = re.findall(r"/\*\*(.*?)\*/", content, re.DOTALL)
            title = jsdoc[0].splitlines()[0] if jsdoc else ""
            summary = jsdoc[0] if jsdoc else ""
        else:
            title = ""
            summary = ""
        return {"title": title, "summary": summary}

    """
    Ingests and parses documentation from URLs, local files, and API endpoints (OpenAPI/Swagger).
    """

    def ingest_url(self, url: str) -> Optional[str]:
        """Fetches and returns main content from a webpage or raw file URL."""
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"Failed to fetch URL {url}: {e}")
            return None

    def ingest_local_file(self, file_path: str) -> Optional[str]:
        """Reads and returns content from a local file (README, docs, config)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")
            return None

    def ingest_directory(
        self,
        dir_path: str,
        exts: List[str] = [".md", ".rst", ".txt", ".yaml", ".yml", ".json"],
    ) -> Dict[str, str]:
        """Reads all documentation files in a directory and returns their contents."""
        docs = {}
        p = pathlib.Path(dir_path)
        for file in p.rglob("*"):
            if file.suffix.lower() in exts:
                try:
                    docs[str(file)] = file.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"Failed to read {file}: {e}")
        return docs

    def ingest_openapi(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetches and parses OpenAPI/Swagger JSON/YAML from an API endpoint."""
        try:
            resp = requests.get(endpoint, timeout=10)
            resp.raise_for_status()
            if endpoint.endswith(".yaml") or endpoint.endswith(".yml"):
                return yaml.safe_load(resp.text)
            else:
                return resp.json()
        except Exception as e:
            print(f"Failed to fetch OpenAPI spec from {endpoint}: {e}")
            return None

    def extract_metadata(self, doc_content: str) -> Dict[str, Any]:
        """Extracts metadata (title, summary, endpoints, etc.) from documentation content."""
        # Simple demo: extract first header and summary
        lines = doc_content.splitlines()
        title = lines[0] if lines else ""
        summary = ""
        for line in lines[1:]:
            if line.strip():
                summary = line.strip()
                break
        return {"title": title, "summary": summary}

    def enrich_with_registry(
        self, component_name: str, doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrates extracted metadata with component registry info."""
        # Placeholder for registry integration
        enriched = doc_metadata.copy()
        enriched["component"] = component_name
        return enriched
