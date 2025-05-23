"""
Microsoft AI Integration for Tygent

This module provides optimized integration with Microsoft's AI services, including Azure OpenAI and Semantic Kernel.
"""
from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
from ..dag import DAG
from ..nodes import BaseNode, LLMNode
from ..scheduler import Scheduler

class MicrosoftAINode(LLMNode):
    """A node that interacts with Microsoft AI services."""
    
    def __init__(
        self, 
        name: str,
        client: Any,
        deployment_id: Optional[str] = None,
        prompt_template: str = "",
        dependencies: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize a Microsoft AI node.
        
        Args:
            name: The name of the node
            client: Microsoft AI client instance
            deployment_id: Azure OpenAI deployment ID
            prompt_template: Template string for the prompt
            dependencies: List of node names this node depends on
        """
        super().__init__(name, dependencies=dependencies)
        self.client = client
        self.deployment_id = deployment_id
        self.prompt_template = prompt_template
        self.kwargs = kwargs
    
    async def execute(self, inputs: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """
        Execute the node by calling the Microsoft AI service.
        
        Args:
            inputs: Input values for the node
            context: Execution context
            
        Returns:
            The response from the Microsoft AI service
        """
        context = context or {}
        prompt = self._format_prompt(inputs, context)
        
        # Execute the Microsoft AI call
        try:
            result = await self._call_microsoft_ai(prompt)
            return result
        except Exception as e:
            self.logger.error(f"Error executing Microsoft AI node: {e}")
            raise
    
    def _format_prompt(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Format the prompt template with inputs and context."""
        # Combine inputs and context for template formatting
        format_dict = {**inputs, **context}
        
        # Use the provided prompt template or a direct input if no template
        if self.prompt_template:
            return self.prompt_template.format(**format_dict)
        elif "prompt" in inputs:
            return inputs["prompt"]
        else:
            return str(inputs)
    
    async def _call_microsoft_ai(self, prompt: str) -> Any:
        """Call the Microsoft AI service with the prepared prompt."""
        # For Azure OpenAI client
        if hasattr(self.client, "get_completions"):
            response = await self.client.get_completions(
                deployment_id=self.deployment_id or self.kwargs.get("deployment_id", ""),
                prompt=prompt,
                max_tokens=self.kwargs.get("max_tokens", 500),
                temperature=self.kwargs.get("temperature", 0.7),
                **{k: v for k, v in self.kwargs.items() if k not in ["max_tokens", "temperature"]}
            )
            if hasattr(response, "choices") and len(response.choices) > 0:
                return response.choices[0].text
            return response
            
        # For Semantic Kernel
        elif hasattr(self.client, "invoke_prompt") or hasattr(self.client, "invoke_semantic_function"):
            if hasattr(self.client, "invoke_prompt"):
                return await self.client.invoke_prompt(prompt, **self.kwargs)
            else:
                return await self.client.invoke_semantic_function(prompt, **self.kwargs)
                
        # Generic async call for other client types
        else:
            return await self.client(prompt, **self.kwargs)


class MicrosoftAIIntegration:
    """Integration with Microsoft AI services for optimized execution."""
    
    def __init__(self, client: Any, deployment_id: Optional[str] = None, **kwargs):
        """
        Initialize the Microsoft AI integration.
        
        Args:
            client: Microsoft AI client to use
            deployment_id: Azure OpenAI deployment ID
            **kwargs: Additional configuration options
        """
        self.client = client
        self.deployment_id = deployment_id
        self.config = kwargs
        self.dag = DAG()
        self.scheduler = Scheduler()
    
    def create_node(
        self, 
        name: str, 
        prompt_template: str = "", 
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> MicrosoftAINode:
        """
        Create a Microsoft AI node.
        
        Args:
            name: The name of the node
            prompt_template: Template string for the prompt
            dependencies: List of node names this node depends on
            **kwargs: Additional node configuration
            
        Returns:
            The created Microsoft AI node
        """
        node = MicrosoftAINode(
            name=name,
            client=self.client,
            deployment_id=self.deployment_id,
            prompt_template=prompt_template,
            dependencies=dependencies,
            **kwargs
        )
        self.dag.add_node(node)
        return node
    
    def optimize(self, constraints: Optional[Dict[str, Any]] = None) -> 'MicrosoftAIIntegration':
        """
        Apply optimization settings to the DAG.
        
        Args:
            constraints: Resource constraints to apply
            
        Returns:
            Self for chaining
        """
        constraints = constraints or {}
        
        # Configure the scheduler with constraints
        scheduler_options = {
            "max_parallel_nodes": constraints.get("max_parallel_nodes", 5),
            "max_execution_time": constraints.get("max_execution_time", 60000),
            "priority_nodes": constraints.get("priority_nodes", []),
        }
        
        self.scheduler.configure(**scheduler_options)
        return self
    
    async def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the DAG with Microsoft AI nodes.
        
        Args:
            inputs: Input values for the execution
            context: Execution context
            
        Returns:
            The execution results
        """
        context = context or {}
        
        # Execute the DAG with the scheduler
        results = await self.scheduler.execute(self.dag, inputs, context)
        return results


class SemanticKernelOptimizer:
    """Optimizer for Microsoft's Semantic Kernel."""
    
    def __init__(self, kernel: Any):
        """
        Initialize the Semantic Kernel optimizer.
        
        Args:
            kernel: Semantic Kernel instance
        """
        self.kernel = kernel
        self.dag = DAG()
        self.scheduler = Scheduler()
        self.plugins = {}
    
    def register_plugin(self, plugin: Any, name: Optional[str] = None) -> None:
        """
        Register a Semantic Kernel plugin with optimized execution.
        
        Args:
            plugin: Semantic Kernel plugin
            name: Optional name for the plugin
        """
        plugin_name = name or getattr(plugin, "name", f"plugin_{len(self.plugins)}")
        self.plugins[plugin_name] = plugin
        
        # Extract functions from the plugin and create nodes
        for func_name in dir(plugin):
            if not func_name.startswith("_") and callable(getattr(plugin, func_name)):
                func = getattr(plugin, func_name)
                if hasattr(func, "is_semantic_function") or hasattr(func, "is_native_function"):
                    node_name = f"{plugin_name}_{func_name}"
                    self._create_function_node(node_name, plugin, func_name)
    
    def _create_function_node(self, name: str, plugin: Any, func_name: str) -> BaseNode:
        """Create a node for a Semantic Kernel function."""
        async def execute_function(inputs, context):
            # Extract SK function
            func = getattr(plugin, func_name)
            
            # Prepare arguments based on function type
            if hasattr(func, "is_semantic_function"):
                # Semantic function typically expects a string input
                input_value = inputs.get("input", "")
                if isinstance(input_value, dict):
                    # Convert dict to string for semantic functions
                    input_value = str(input_value)
                return await func(input_value)
            else:
                # Native function can take more structured inputs
                return await func(**inputs)
        
        # Create and add node
        node = BaseNode(name)
        node.execute = execute_function
        self.dag.add_node(node)
        return node
    
    def create_plan(self, plan_description: str, **kwargs) -> 'SemanticKernelOptimizer':
        """
        Create an optimized execution plan based on a description.
        
        Args:
            plan_description: Natural language description of the plan
            **kwargs: Additional planning options
            
        Returns:
            Self for chaining
        """
        # In a real implementation, this would extract a plan from the description
        # and create the necessary nodes and dependencies
        
        # For now, we'll just return self
        return self
        
    def optimize(self, constraints: Optional[Dict[str, Any]] = None) -> 'SemanticKernelOptimizer':
        """
        Apply optimization settings to the execution plan.
        
        Args:
            constraints: Resource constraints to apply
            
        Returns:
            Self for chaining
        """
        constraints = constraints or {}
        
        # Configure the scheduler with constraints
        scheduler_options = {
            "max_parallel_nodes": constraints.get("max_parallel_nodes", 5),
            "max_execution_time": constraints.get("max_execution_time", 60000),
            "priority_nodes": constraints.get("priority_nodes", []),
        }
        
        self.scheduler.configure(**scheduler_options)
        return self
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the optimized plan.
        
        Args:
            inputs: Input values for the execution
            
        Returns:
            The execution results
        """
        # Execute the DAG with the scheduler
        results = await self.scheduler.execute(self.dag, inputs)
        return results