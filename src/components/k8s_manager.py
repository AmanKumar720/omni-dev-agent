"""
k8s_manager.py
Kubernetes integration for omni-dev-agent
"""

from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException
import yaml


class K8sManager:
    """
    Usage Example:
    --------------
    from k8s_manager import K8sManager
    k8s = K8sManager()
    k8s.deploy_resource('deployment.yaml', namespace='default')
    status = k8s.get_resource_status('deployment', 'my-app', namespace='default')
    print(status)
    k8s.delete_resource('deployment', 'my-app', namespace='default')
    """

    def __init__(self, kubeconfig_path=None):
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_kube_config()
            self.api_client = client.ApiClient()
        except Exception as e:
            print(f"Error loading kubeconfig: {e}")

    def deploy_resource(self, yaml_path, namespace="default"):
        """Deploy resources from a YAML file to the specified namespace."""
        try:
            with open(yaml_path) as f:
                docs = list(yaml.safe_load_all(f))
            for doc in docs:
                utils.create_from_dict(self.api_client, doc, namespace=namespace)
        except FileNotFoundError:
            print(f"YAML file not found: {yaml_path}")
        except ApiException as e:
            print(f"Error deploying resource: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def get_resource_status(self, kind, name, namespace="default"):
        """Get status of a resource by kind, name, and namespace."""
        try:
            kind = kind.lower()
            if kind == "deployment":
                api = client.AppsV1Api()
                return api.read_namespaced_deployment_status(name, namespace)
            elif kind == "pod":
                api = client.CoreV1Api()
                return api.read_namespaced_pod_status(name, namespace)
            elif kind == "service":
                api = client.CoreV1Api()
                return api.read_namespaced_service_status(name, namespace)
            elif kind == "statefulset":
                api = client.AppsV1Api()
                return api.read_namespaced_stateful_set_status(name, namespace)
            elif kind == "job":
                api = client.BatchV1Api()
                return api.read_namespaced_job_status(name, namespace)
            else:
                print(f"Unsupported kind: {kind}")
                return None
        except ApiException as e:
            print(f"Error getting status: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def delete_resource(self, kind, name, namespace="default"):
        """Delete a resource by kind, name, and namespace."""
        try:
            kind = kind.lower()
            if kind == "deployment":
                api = client.AppsV1Api()
                api.delete_namespaced_deployment(name, namespace)
            elif kind == "pod":
                api = client.CoreV1Api()
                api.delete_namespaced_pod(name, namespace)
            elif kind == "service":
                api = client.CoreV1Api()
                api.delete_namespaced_service(name, namespace)
            elif kind == "statefulset":
                api = client.AppsV1Api()
                api.delete_namespaced_stateful_set(name, namespace)
            elif kind == "job":
                api = client.BatchV1Api()
                api.delete_namespaced_job(name, namespace)
            else:
                print(f"Unsupported kind: {kind}")
        except ApiException as e:
            print(f"Error deleting resource: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
