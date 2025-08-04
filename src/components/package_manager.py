import requests
from .component_registry.registry import get_component_metadata


class PackageManager:
    def __init__(self):
        print("Initializing Package Manager...")

    def assign_project(self, project_details, agents, human_in_loop=True):
        """
        Assigns a project to the best-fit agent/component using AI or rule-based logic.
        If human_in_loop is True, presents recommendation for human approval.
        If False, assigns automatically.
        """
        # Example: agents is a list of dicts with 'name', 'skills', 'score', etc.
        # project_details is a dict with 'type', 'requirements', etc.
        # For demo, pick agent with highest score matching project type
        candidates = [
            a for a in agents if project_details["type"] in a.get("skills", [])
        ]
        if not candidates:
            print("No suitable agent found for project type.")
            return None
        best_agent = max(candidates, key=lambda a: a.get("score", 0))
        if human_in_loop:
            print(
                f"Recommended agent for project '{project_details['name']}': {best_agent['name']} (score: {best_agent['score']})"
            )
            choice = input("Assign this agent? (y/n): ")
            if choice.lower() == "y":
                print(f"Project assigned to {best_agent['name']}.")
                return best_agent
            else:
                print("Assignment cancelled by human.")
                return None
        else:
            print(
                f"Project '{project_details['name']}' automatically assigned to {best_agent['name']} (score: {best_agent['score']})"
            )
            return best_agent

    def self_heal_component(
        self,
        component_name: str,
        language: str = "python",
        project_license: str = "MIT",
    ):
        """
        Detects unhealthy, outdated, or vulnerable components and suggests or triggers fixes.
        """
        enriched = self.enrich_metadata(component_name, language)
        health_score, health_label = self.score_health(enriched)
        license_ok, license_msg = self.check_license_compatibility(
            enriched.get("license"), project_license
        )
        vulns = enriched.get("vulnerabilities", [])
        actions = []
        if health_score < 60:
            actions.append("Update to latest version or consider alternatives.")
        if not license_ok:
            actions.append(f"License issue: {license_msg}")


class PackageManager:
    def __init__(self):
        print("Initializing Package Manager...")

    def score_health(self, enriched):
        """
        Scores component health based on popularity, update frequency, vulnerabilities, and community.
        Returns a score (0-100) and a health label.
        """
        score = 100
        # Popularity (dependent repos, stars, rank)
        if enriched.get("dependent_repos_count", 0) < 10:
            score -= 20
        if enriched.get("stars", 0) and enriched.get("stars", 0) < 50:
            score -= 10
        if enriched.get("rank", 0) and enriched.get("rank", 0) > 10000:
            score -= 10
        # Update frequency
        if enriched.get("latest_release_published_at"):
            from datetime import datetime, timezone

            try:
                last_release = datetime.fromisoformat(
                    enriched["latest_release_published_at"].replace("Z", "+00:00")
                )
                days_since = (datetime.now(timezone.utc) - last_release).days
                if days_since > 365:
                    score -= 20
            except Exception:
                pass
        # Vulnerabilities
        vulns = enriched.get("vulnerabilities", [])
        if isinstance(vulns, list) and len(vulns) > 0:
            score -= 30
        # Community
        if (
            enriched.get("community", 0)
            and isinstance(enriched["community"], int)
            and enriched["community"] < 10
        ):
            score -= 10
        # Clamp score
        score = max(0, min(100, score))
        if score >= 80:
            label = "Excellent"
        elif score >= 60:
            label = "Good"
        elif score >= 40:
            label = "Fair"
        else:
            label = "Poor"
        return score, label

    def check_license_compatibility(self, component_license, project_license):
        """
        Checks if the component's license is compatible with the project's license.
        Returns True/False and a message.
        """
        # Simple mapping for demo; extend with SPDX or license DB for real use
        compatible = True
        message = "Compatible."
        if not component_license or not project_license:
            compatible = False
            message = "License info missing."
        elif project_license.lower() == "mit":
            if component_license.lower() in ["gpl", "agpl", "lgpl", "epl"]:
                compatible = False
                message = f"{component_license} is not compatible with MIT."
        elif project_license.lower() == "gpl":
            if component_license.lower() not in ["gpl", "agpl", "lgpl"]:
                compatible = False
                message = f"{component_license} is not compatible with GPL."
        # Add more rules as needed
        return compatible, message

    def contextualize_component(
        self,
        component_name: str,
        language: str = "python",
        project_license: str = "MIT",
    ):
        """
        Returns detailed context for a component: purpose, environment compatibility, licensing, quality, and cost.
        Enriches metadata from external sources.
        """
        enriched = self.enrich_metadata(component_name, language)
        if enriched:
            purpose = enriched.get("purpose", "No purpose description available.")
            description = enriched.get("description", "No description available.")
            environment = enriched.get(
                "environment", "Unknown environment compatibility."
            )
            license_info = enriched.get("license", "License info not available.")
            quality = enriched.get("quality", "Quality/reliability info not available.")
            known_issues = enriched.get("known_issues", "No known issues listed.")
            vulnerabilities = enriched.get("vulnerabilities", "No vulnerability info.")
            community = enriched.get(
                "community", "Community support info not available."
            )
            cost = enriched.get("cost", "Cost info not available.")
            version = enriched.get("version", "Unknown")
            stars = enriched.get("stars", "N/A")
            forks = enriched.get("forks", "N/A")
            open_issues = enriched.get("open_issues", "N/A")
            health_score, health_label = self.score_health(enriched)
            license_ok, license_msg = self.check_license_compatibility(
                license_info, project_license
            )
            print(
                f"Component: {component_name}\nPurpose: {purpose}\nDescription: {description}\nEnvironment: {environment}\nLicense: {license_info} ({license_msg})\nQuality/Reliability: {quality}\nKnown Issues: {known_issues}\nVulnerabilities: {vulnerabilities}\nCommunity Support: {community}\nCost: {cost}\nVersion: {version}\nGitHub Stars: {stars}\nForks: {forks}\nOpen Issues: {open_issues}\nHealth Score: {health_score} ({health_label})"
            )
            return {
                "component": component_name,
                "purpose": purpose,
                "description": description,
                "environment": environment,
                "license": license_info,
                "license_compatible": license_ok,
                "license_message": license_msg,
                "quality": quality,
                "known_issues": known_issues,
                "vulnerabilities": vulnerabilities,
                "community": community,
                "cost": cost,
                "version": version,
                "stars": stars,
                "forks": forks,
                "open_issues": open_issues,
                "health_score": health_score,
                "health_label": health_label,
            }
        else:
            print(
                f"Component {component_name} not found in registry or external sources."
            )
            return None

    def self_heal_component(
        self,
        component_name: str,
        language: str = "python",
        project_license: str = "MIT",
    ):
        """
        Detects unhealthy, outdated, or vulnerable components and suggests or triggers fixes.
        """
        enriched = self.enrich_metadata(component_name, language)
        health_score, health_label = self.score_health(enriched)
        license_ok, license_msg = self.check_license_compatibility(
            enriched.get("license"), project_license
        )
        vulns = enriched.get("vulnerabilities", [])
        actions = []
        if health_score < 60:
            actions.append("Update to latest version or consider alternatives.")
        if not license_ok:
            actions.append(f"License issue: {license_msg}")
        if isinstance(vulns, list) and len(vulns) > 0:
            actions.append("Vulnerabilities detected: consider patching or replacing.")
        if not actions:
            actions.append("No action needed. Component is healthy and compatible.")
        print(
            f"Self-heal analysis for {component_name}:\nHealth: {health_label} ({health_score})\nLicense: {enriched.get('license')} ({license_msg})\nVulnerabilities: {vulns}\nRecommended actions: {actions}"
        )
        return {
            "component": component_name,
            "health": health_label,
            "health_score": health_score,
            "license": enriched.get("license"),
            "license_compatible": license_ok,
            "license_message": license_msg,
            "vulnerabilities": vulns,
            "actions": actions,
        }

    def assign_project(self, project_details, agents, human_in_loop=True):
        """
        Assigns a project to the best-fit agent/component using AI or rule-based logic.
        If human_in_loop is True, presents recommendation for human approval.
        If False, assigns automatically.
        """
        # Example: agents is a list of dicts with 'name', 'skills', 'score', etc.
        # project_details is a dict with 'type', 'requirements', etc.
        # For demo, pick agent with highest score matching project type
        candidates = [
            a for a in agents if project_details["type"] in a.get("skills", [])
        ]
        if not candidates:
            print("No suitable agent found for project type.")
            return None
        best_agent = max(candidates, key=lambda a: a.get("score", 0))
        if human_in_loop:
            print(
                f"Recommended agent for project '{project_details['name']}': {best_agent['name']} (score: {best_agent['score']})"
            )
            choice = input("Assign this agent? (y/n): ")
            if choice.lower() == "y":
                print(f"Project assigned to {best_agent['name']}.")
                return best_agent
            else:
                print("Assignment cancelled by human.")
                return None
        else:
            print(
                f"Project '{project_details['name']}' automatically assigned to {best_agent['name']} (score: {best_agent['score']})"
            )
            return best_agent
        # Vulnerabilities
        vulns = enriched.get("vulnerabilities", [])
        if isinstance(vulns, list) and len(vulns) > 0:
            score -= 30
        # Community
        if (
            enriched.get("community", 0)
            and isinstance(enriched["community"], int)
            and enriched["community"] < 10
        ):
            score -= 10
        # Clamp score
        score = max(0, min(100, score))
        if score >= 80:
            label = "Excellent"
        elif score >= 60:
            label = "Good"
        elif score >= 40:
            label = "Fair"
        else:
            label = "Poor"
        return score, label

    def check_license_compatibility(self, component_license, project_license):
        """
        Checks if the component's license is compatible with the project's license.
        Returns True/False and a message.
        """
        # Simple mapping for demo; extend with SPDX or license DB for real use
        compatible = True
        message = "Compatible."
        if not component_license or not project_license:
            compatible = False
            message = "License info missing."
        elif project_license.lower() == "mit":
            if component_license.lower() in ["gpl", "agpl", "lgpl", "epl"]:
                compatible = False
                message = f"{component_license} is not compatible with MIT."
        elif project_license.lower() == "gpl":
            if component_license.lower() not in ["gpl", "agpl", "lgpl"]:
                compatible = False
                message = f"{component_license} is not compatible with GPL."
        # Add more rules as needed
        return compatible, message

    def enrich_metadata(self, component_name: str, language: str = "python"):
        """
        Fetches metadata from external sources (PyPI, npm, GitHub) and merges with registry metadata.
        """
        metadata = get_component_metadata(component_name) or {}
        external = {}
        if language == "python":
            url = f"https://pypi.org/pypi/{component_name}/json"
            resp = requests.get(url)
            if resp.status_code == 200:
                info = resp.json().get("info", {})
                external.update(
                    {
                        "description": info.get("summary"),
                        "license": info.get("license"),
                        "version": info.get("version"),
                        "home_page": info.get("home_page"),
                        "author": info.get("author"),
                        "community": info.get("author_email"),
                    }
                )
        elif language == "node":
            url = f"https://registry.npmjs.org/{component_name}"
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                external.update(
                    {
                        "description": data.get("description"),
                        "license": data.get("license"),
                        "version": data.get("dist-tags", {}).get("latest"),
                        "home_page": data.get("homepage"),
                        "author": data.get("author", {}).get("name"),
                        "community": data.get("maintainers", []),
                    }
                )
        elif language == "ruby":
            url = f"https://rubygems.org/api/v1/gems/{component_name}.json"
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                external.update(
                    {
                        "description": data.get("info"),
                        "license": data.get("licenses", [""])[0],
                        "version": data.get("version"),
                        "home_page": data.get("homepage_uri"),
                        "author": data.get("authors"),
                        "community": data.get("downloads"),
                    }
                )
        elif language == "php":
            url = f"https://packagist.org/search.json?q={component_name}"
            resp = requests.get(url)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    pkg = results[0]
                    external.update(
                        {
                            "description": pkg.get("description"),
                            "license": (
                                pkg.get("license", [""])[0]
                                if pkg.get("license")
                                else ""
                            ),
                            "version": pkg.get("version"),
                            "home_page": pkg.get("homepage"),
                            "author": pkg.get("name"),
                            "community": pkg.get("downloads"),
                        }
                    )
        elif language == "r":
            url = f"https://crandb.r-pkg.org/{component_name}"
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                external.update(
                    {
                        "description": data.get("Title"),
                        "license": data.get("License"),
                        "version": data.get("Version"),
                        "home_page": data.get("URL"),
                        "author": data.get("Author"),
                        "community": data.get("Maintainer"),
                    }
                )
        elif language == "java":
            url = f"https://search.maven.org/solrsearch/select?q={component_name}&rows=1&wt=json"
            resp = requests.get(url)
            if resp.status_code == 200:
                docs = resp.json().get("response", {}).get("docs", [])
                if docs:
                    doc = docs[0]
                    external.update(
                        {
                            "description": doc.get("latestVersion"),
                            "license": doc.get("license", ""),
                            "version": doc.get("latestVersion"),
                            "home_page": doc.get("repositoryId", ""),
                            "author": doc.get("g"),
                            "community": doc.get("id"),
                        }
                    )
        # GitHub enrichment (if home_page is a GitHub repo)
        repo_url = external.get("home_page", "")
        if repo_url and "github.com" in repo_url:
            api_url = repo_url.replace(
                "https://github.com/", "https://api.github.com/repos/"
            )
            resp = requests.get(api_url)
            if resp.status_code == 200:
                gh = resp.json()
                external.update(
                    {
                        "stars": gh.get("stargazers_count"),
                        "forks": gh.get("forks_count"),
                        "open_issues": gh.get("open_issues_count"),
                        "community": gh.get("subscribers_count"),
                    }
                )

        # OSV vulnerability enrichment
        osv_url = f"https://api.osv.dev/v1/query"
        osv_payload = {
            "package": {"name": component_name, "ecosystem": language.capitalize()}
        }
        try:
            osv_resp = requests.post(osv_url, json=osv_payload)
            if osv_resp.status_code == 200:
                vulns = osv_resp.json().get("vulns", [])
                external["vulnerabilities"] = (
                    [v.get("summary") for v in vulns]
                    if vulns
                    else "No known vulnerabilities."
                )
        except Exception:
            external["vulnerabilities"] = "Vulnerability check failed."

        # libraries.io enrichment (popularity, platform, etc.)
        libio_url = (
            f"https://libraries.io/api/search?q={component_name}&platforms={language}"
        )
        try:
            libio_resp = requests.get(libio_url)
            if libio_resp.status_code == 200:
                results = libio_resp.json()
                if results:
                    lib = results[0]
                    external.update(
                        {
                            "platform": lib.get("platform"),
                            "dependent_repos_count": lib.get("dependent_repos_count"),
                            "rank": lib.get("rank"),
                            "latest_release_published_at": lib.get(
                                "latest_release_published_at"
                            ),
                        }
                    )
        except Exception:
            pass
        # Merge external with registry metadata
        enriched = {**metadata, **external}
        return enriched

    def contextualize_component(
        self,
        component_name: str,
        language: str = "python",
        project_license: str = "MIT",
    ):
        """
        Returns detailed context for a component: purpose, environment compatibility, licensing, quality, and cost.
        Enriches metadata from external sources.
        """
        enriched = self.enrich_metadata(component_name, language)
        if enriched:
            purpose = enriched.get("purpose", "No purpose description available.")
            description = enriched.get("description", "No description available.")
            environment = enriched.get(
                "environment", "Unknown environment compatibility."
            )
            license_info = enriched.get("license", "License info not available.")
            quality = enriched.get("quality", "Quality/reliability info not available.")
            known_issues = enriched.get("known_issues", "No known issues listed.")
            vulnerabilities = enriched.get("vulnerabilities", "No vulnerability info.")
            community = enriched.get(
                "community", "Community support info not available."
            )
            cost = enriched.get("cost", "Cost info not available.")
            version = enriched.get("version", "Unknown")
            stars = enriched.get("stars", "N/A")
            forks = enriched.get("forks", "N/A")
            open_issues = enriched.get("open_issues", "N/A")
            health_score, health_label = self.score_health(enriched)
            license_ok, license_msg = self.check_license_compatibility(
                license_info, project_license
            )
            print(
                f"Component: {component_name}\nPurpose: {purpose}\nDescription: {description}\nEnvironment: {environment}\nLicense: {license_info} ({license_msg})\nQuality/Reliability: {quality}\nKnown Issues: {known_issues}\nVulnerabilities: {vulnerabilities}\nCommunity Support: {community}\nCost: {cost}\nVersion: {version}\nGitHub Stars: {stars}\nForks: {forks}\nOpen Issues: {open_issues}\nHealth Score: {health_score} ({health_label})"
            )
            return {
                "component": component_name,
                "purpose": purpose,
                "description": description,
                "environment": environment,
                "license": license_info,
                "license_compatible": license_ok,
                "license_message": license_msg,
                "quality": quality,
                "known_issues": known_issues,
                "vulnerabilities": vulnerabilities,
                "community": community,
                "cost": cost,
                "version": version,
                "stars": stars,
                "forks": forks,
                "open_issues": open_issues,
                "health_score": health_score,
                "health_label": health_label,
            }
        else:
            print(
                f"Component {component_name} not found in registry or external sources."
            )
            return None

    def install_component(self, component_name: str):
        """
        Simulates installing a component based on its name.
        In a real scenario, this would involve using pip, npm, etc.
        """
        print(f"Attempting to install component: {component_name}")
        metadata = get_component_metadata(component_name)
        if metadata:
            print(f"Found metadata for {component_name}: {metadata.get('description')}")
            dependencies = metadata.get("dependencies", [])
            if dependencies:
                print(f"Component has dependencies: {', '.join(dependencies)}")
                # In a real scenario, this would trigger actual installation commands
                # e.g., run_shell_command(f"pip install {dep}") for each dependency
                print(
                    f" (Placeholder: Installing dependencies: {', '.join(dependencies)})"
                )
            else:
                print(f"No specific dependencies listed for {component_name}.")
            print(f" (Placeholder: Installing {component_name} itself)")
            return {
                "status": "success",
                "message": f"Simulated installation of {component_name}",
            }
        else:
            print(f"Component {component_name} not found in registry. Cannot install.")
            return {
                "status": "failed",
                "message": f"Component {component_name} not found",
            }

    def check_component_version(self, component_name: str, installed_version: str):
        """
        Simulates checking the installed version of a component against the registry.
        """
        print(
            f"Checking version for {component_name} (installed: {installed_version})..."
        )
        metadata = get_component_metadata(component_name)
        if (
            metadata and metadata.get("version") != "latest"
        ):  # Assuming 'latest' means we don't have a specific version to compare
            expected_version = metadata.get("version")
            if installed_version == expected_version:
                print(
                    f"Version for {component_name} is up to date: {installed_version}"
                )
                return {
                    "status": "up_to_date",
                    "message": f"{component_name} is up to date",
                }
            else:
                print(
                    f"Version mismatch for {component_name}: Installed {installed_version}, Expected {expected_version}"
                )
                return {
                    "status": "version_mismatch",
                    "message": f"Version mismatch for {component_name}",
                }
        else:
            print(
                f"No specific version to compare for {component_name} in registry or metadata not found."
            )
            return {
                "status": "no_comparison",
                "message": "No specific version to compare",
            }


if __name__ == "__main__":
    pkg_manager = PackageManager()

    # Example 1: Install a component from the registry
    print("\n--- Installing Supabase ---")
    install_result = pkg_manager.install_component("Supabase")
    print("Install Result:", install_result)

    # Example 2: Install a component not in the registry
    print("\n--- Installing NonExistentComponent ---")
    install_result = pkg_manager.install_component("NonExistentComponent")
    print("Install Result:", install_result)

    # Example 3: Check version of an existing component
    print("\n--- Checking Flask Version ---")
    version_check_result = pkg_manager.check_component_version("Flask", "3.0.0")
    print("Version Check Result:", version_check_result)

    # Example 4: Check version of a component with 'latest' in registry
    print("\n--- Checking boto3 Version ---")
    version_check_result = pkg_manager.check_component_version("boto3", "1.34.0")
    print("Version Check Result:", version_check_result)

    # Example 5: Contextualize a component's purpose and functionality
    print("\n--- Contextualizing Supabase ---")
    context_result = pkg_manager.contextualize_component("Supabase")
