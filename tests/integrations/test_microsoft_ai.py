"""
Tests for Microsoft AI integration with Tygent.
"""
import asyncio
import pytest
import unittest
from unittest.mock import MagicMock, patch
from tygent.integrations.microsoft_ai import MicrosoftAINode, MicrosoftAIIntegration, SemanticKernelOptimizer

# Mock response from Azure OpenAI
class MockAzureOpenAIResponse:
    def __init__(self, text):
        self.choices = [MagicMock(text=text)]

# Mock Azure OpenAI client
class MockAzureOpenAIClient:
    async def get_completions(self, **kwargs):
        # Return different responses based on the prompt
        prompt = kwargs.get("prompt", "")
        deployment_id = kwargs.get("deployment_id", "")
        
        if "market overview" in prompt.lower():
            return MockAzureOpenAIResponse("The renewable energy market in Southeast Asia is growing rapidly...")
        elif "trends" in prompt.lower():
            return MockAzureOpenAIResponse("1. Increased solar adoption\n2. Wind energy expansion\n3. Battery storage innovations")
        elif "competitors" in prompt.lower():
            return MockAzureOpenAIResponse("Top competitors include: 1. SunPower Corp, 2. First Solar, 3. NextEra Energy")
        elif "regulatory" in prompt.lower():
            return MockAzureOpenAIResponse("Key regulations include renewable energy targets and feed-in tariffs.")
        elif "growth" in prompt.lower():
            return MockAzureOpenAIResponse("Growth opportunities: 1. Microgrids for island communities, 2. Solar-plus-storage solutions")
        elif "strategy" in prompt.lower():
            return MockAzureOpenAIResponse("Market entry strategy: Start with partnerships to establish local presence.")
        else:
            return MockAzureOpenAIResponse("Generic response for: " + prompt)

# Mock Semantic Kernel function
class MockSemanticFunction:
    def __init__(self, name):
        self.name = name
        self.is_semantic_function = True
        
    async def __call__(self, input_str):
        return f"Processed {self.name}: {input_str}"

# Mock Semantic Kernel plugin
class MockSKPlugin:
    def __init__(self):
        self.name = "TextAnalysis"
        self.sentiment_analysis = MockSemanticFunction("sentiment_analysis")
        self.summarize = MockSemanticFunction("summarize")

# Mock Semantic Kernel
class MockSemanticKernel:
    def __init__(self):
        self.functions = {}
        
    def add_semantic_function(self, *args, **kwargs):
        return MockSemanticFunction("custom_function")

class TestMicrosoftAINode(unittest.TestCase):
    """Tests for MicrosoftAINode class."""
    
    def setUp(self):
        self.client = MockAzureOpenAIClient()
        self.node = MicrosoftAINode(
            name="test_node",
            client=self.client,
            deployment_id="gpt-4",
            prompt_template="Analysis of {topic}"
        )
    
    def test_initialization(self):
        """Test node initialization."""
        self.assertEqual(self.node.name, "test_node")
        self.assertEqual(self.node.deployment_id, "gpt-4")
        self.assertEqual(self.node.prompt_template, "Analysis of {topic}")
        
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test node execution."""
        result = await self.node.execute({"topic": "market trends"})
        self.assertIn("solar", result.lower())
        
    @pytest.mark.asyncio
    async def test_format_prompt(self):
        """Test prompt formatting."""
        formatted = self.node._format_prompt({"topic": "competitors"}, {})
        self.assertEqual(formatted, "Analysis of competitors")
        
        # Test with context
        formatted = self.node._format_prompt({"topic": "basic"}, {"additional": "context"})
        self.assertEqual(formatted, "Analysis of basic")

class TestMicrosoftAIIntegration(unittest.TestCase):
    """Tests for MicrosoftAIIntegration class."""
    
    def setUp(self):
        self.client = MockAzureOpenAIClient()
        self.integration = MicrosoftAIIntegration(self.client, "gpt-4")
    
    def test_initialization(self):
        """Test integration initialization."""
        self.assertIsNotNone(self.integration.dag)
        self.assertIsNotNone(self.integration.scheduler)
        self.assertEqual(self.integration.deployment_id, "gpt-4")
        
    def test_add_node(self):
        """Test adding a node to the integration."""
        node = self.integration.addNode(
            name="market_overview",
            prompt_template="Provide a market overview of {industry} in {region}",
            dependencies=["industry_research"]
        )
        
        self.assertEqual(node.name, "market_overview")
        self.assertEqual(node.prompt_template, "Provide a market overview of {industry} in {region}")
        self.assertEqual(node.dependencies, ["industry_research"])
        
    def test_optimize(self):
        """Test optimization settings."""
        self.integration.optimize({
            "maxParallelCalls": 3,
            "maxExecutionTime": 30000,
            "priorityNodes": ["market_overview"]
        })
        
        # Verify the settings were applied
        self.assertEqual(self.integration.scheduler.max_parallel_nodes, 3)
        self.assertEqual(self.integration.scheduler.max_execution_time, 30000)
        self.assertEqual(self.integration.scheduler.priority_nodes, ["market_overview"])
        
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test execution of the integration DAG."""
        # Add nodes to test
        self.integration.addNode(
            name="market_overview",
            prompt_template="Provide a market overview of {industry} in {region}",
            dependencies=[]
        )
        
        self.integration.addNode(
            name="market_trends",
            prompt_template="What are the top trends in {industry} in {region} that match this overview: {market_overview}",
            dependencies=["market_overview"]
        )
        
        # Execute the DAG
        results = await self.integration.execute({
            "industry": "renewable energy", 
            "region": "Southeast Asia"
        })
        
        # Check that both nodes were executed
        self.assertIn("market_overview", results)
        self.assertIn("market_trends", results)
        
        # Check result content
        self.assertIn("renewable", results["market_overview"].lower())
        self.assertIn("solar", results["market_trends"].lower())

class TestSemanticKernelOptimizer(unittest.TestCase):
    """Tests for SemanticKernelOptimizer class."""
    
    def setUp(self):
        self.kernel = MockSemanticKernel()
        self.plugin = MockSKPlugin()
        self.optimizer = SemanticKernelOptimizer(self.kernel)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.dag)
        self.assertIsNotNone(self.optimizer.scheduler)
        self.assertEqual(self.optimizer.kernel, self.kernel)
        self.assertEqual(len(self.optimizer.plugins), 0)
        
    def test_register_plugin(self):
        """Test registering a Semantic Kernel plugin."""
        self.optimizer.registerPlugin(self.plugin, "text_analysis")
        
        # Verify the plugin was registered
        self.assertIn("text_analysis", self.optimizer.plugins)
        self.assertEqual(self.optimizer.plugins["text_analysis"], self.plugin)
        
        # Verify that nodes were created for the plugin functions
        self.assertEqual(len(self.optimizer.dag.nodes), 2)  # One for each function
        
    def test_create_plan(self):
        """Test creating a plan."""
        # Register plugin first
        self.optimizer.registerPlugin(self.plugin)
        
        # Create a plan
        result = self.optimizer.createPlan("Analyze sentiment and provide recommendations")
        
        # Verify the result is the optimizer itself (for chaining)
        self.assertEqual(result, self.optimizer)
        
    def test_optimize(self):
        """Test optimization settings."""
        self.optimizer.optimize({
            "maxParallelCalls": 2,
            "maxExecutionTime": 10000,
            "priorityNodes": ["TextAnalysis_sentiment_analysis"]
        })
        
        # Verify the settings were applied
        self.assertEqual(self.optimizer.scheduler.max_parallel_nodes, 2)
        self.assertEqual(self.optimizer.scheduler.max_execution_time, 10000)
        self.assertEqual(self.optimizer.scheduler.priority_nodes, ["TextAnalysis_sentiment_analysis"])
        
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test execution of the optimizer."""
        # Register the plugin and create function nodes
        self.optimizer.registerPlugin(self.plugin)
        
        # Execute with input
        results = await self.optimizer.execute({
            "input": "The product is excellent but the customer service needs improvement."
        })
        
        # Check the results
        # Both functions should have been called
        self.assertIn("TextAnalysis_sentiment_analysis", results)
        self.assertIn("TextAnalysis_summarize", results)
        
        # Check result content
        self.assertIn("sentiment_analysis", results["TextAnalysis_sentiment_analysis"])
        self.assertIn("summarize", results["TextAnalysis_summarize"])


if __name__ == "__main__":
    unittest.main()