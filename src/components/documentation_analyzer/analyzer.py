# src/components/documentation_analyzer/analyzer.py

import default_api


def analyze_documentation(component_name: str, documentation_source: str = None):
    """
    Analyzes documentation content for a given component.

    Args:
        component_name (str): The name of the component.
        documentation_source (str, optional): A URL or file path to the documentation.
                                            If None, the agent would attempt to search.
    """
    print(f"\nAnalyzing documentation for component: {component_name}")
    doc_content = ""
    key_terms = []
    integration_steps_summary = (
        "No documentation content provided or found for detailed analysis."
    )

    if documentation_source:
        if documentation_source.startswith("http"):  # Assume URL
            print(f"Attempting to fetch documentation from URL: {documentation_source}")
            try:
                fetch_result = default_api.web_fetch(
                    prompt=f"Get content from {documentation_source}"
                )
                if fetch_result and "output" in fetch_result:
                    doc_content = fetch_result["output"]
                    print(
                        "Documentation fetched successfully (first 200 chars):",
                        doc_content[:200],
                    )
                    # Basic keyword extraction from fetched content
                    key_terms = [
                        term
                        for term in [
                            "API",
                            "Integration",
                            "Setup",
                            "Configuration",
                            "Database",
                            "Authentication",
                            "Real-time",
                            "Tasks",
                            "Asynchronous",
                        ]
                        if term.lower() in doc_content.lower()
                    ]
                    integration_steps_summary = (
                        "Content fetched. Further parsing needed for detailed steps."
                    )
                else:
                    print("Failed to fetch documentation from URL.")
            except Exception as e:
                print(f"Error fetching documentation: {e}")
        else:  # Assume local file path
            print(
                f"Attempting to read documentation from local file: {documentation_source}"
            )
            try:
                read_result = default_api.read_file(absolute_path=documentation_source)
                if read_result and "content" in read_result:
                    doc_content = read_result["content"]
                    print(
                        "Documentation read successfully (first 200 chars):",
                        doc_content[:200],
                    )
                    # Basic keyword extraction from read content
                    key_terms = [
                        term
                        for term in [
                            "API",
                            "Integration",
                            "Setup",
                            "Configuration",
                            "Database",
                            "Authentication",
                            "Real-time",
                            "Tasks",
                            "Asynchronous",
                        ]
                        if term.lower() in doc_content.lower()
                    ]
                    integration_steps_summary = (
                        "Content read. Further parsing needed for detailed steps."
                    )
                else:
                    print("Failed to read documentation from local file.")
            except Exception as e:
                print(f"Error reading documentation: {e}")
    else:
        print("Performing web search for documentation...")
        try:
            search_query = f"{component_name} documentation official site"
            search_results = default_api.google_web_search(query=search_query)
            if search_results and "output" in search_results:
                # Simple heuristic: take the first few URLs that look like official docs
                # In a real scenario, this would involve more sophisticated URL parsing and filtering
                urls = []
                for result in search_results["output"].split("\n"):
                    if result.startswith("http") and "docs" in result:
                        urls.append(result)
                        if len(urls) >= 3:  # Limit to first 3 promising URLs
                            break

                if urls:
                    print(f"Found potential documentation URLs: {urls}")
                    # Attempt to fetch content from the first promising URL
                    for url in urls:
                        try:
                            fetch_result = default_api.web_fetch(
                                prompt=f"Get content from {url}"
                            )
                            if fetch_result and "output" in fetch_result:
                                doc_content = fetch_result["output"]
                                print(
                                    f"Documentation fetched from {url} (first 200 chars):",
                                    doc_content[:200],
                                )
                                # Basic keyword extraction from fetched content
                                key_terms = [
                                    term
                                    for term in [
                                        "API",
                                        "Integration",
                                        "Setup",
                                        "Configuration",
                                        "Database",
                                        "Authentication",
                                        "Real-time",
                                        "Tasks",
                                        "Asynchronous",
                                    ]
                                    if term.lower() in doc_content.lower()
                                ]
                                integration_steps_summary = f"Content fetched from {url}. Further parsing needed for detailed steps."
                                break  # Stop after first successful fetch
                        except Exception as e:
                            print(f"Error fetching documentation from {url}: {e}")
                else:
                    print("No promising documentation URLs found in search results.")
            else:
                print("Web search for documentation failed.")
        except Exception as e:
            print(f"Error during web search for documentation: {e}")

    print(" (Placeholder: Extracting key terms, concepts, and integration steps...)")

    # Return the analysis result
    return {
        "component_name": component_name,
        "key_terms": key_terms,
        "integration_steps_summary": integration_steps_summary,
    }


if __name__ == "__main__":
    # Example usage:
    analysis_result = analyze_documentation("Supabase", "https://supabase.com/docs")
    print("Analysis Result:", analysis_result)

    analysis_result = analyze_documentation("Celery")
    print("Analysis Result:", analysis_result)
