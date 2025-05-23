"""
Tests for Google AI integration with Tygent.
"""
import asyncio
import pytest
import unittest
from unittest.mock import MagicMock, patch
from tygent.integrations.google_ai import GoogleAINode, GoogleAIIntegration, GoogleAIBatchProcessor

# Mock response from Google AI
class MockGoogleAIResponse:
    def __init__(self, text):
        self.text_content = text
        
    def text(self):
        return self.text_content
        
    @property
    def response(self):
        return self

# Mock Google AI model
class MockGoogleAIModel:
    async def generate_content(self, prompt, **kwargs):
        # Return different responses based on the prompt
        if "weather" in prompt.lower():
            return MockGoogleAIResponse("The weather is sunny with temperatures around 75°F.")
        elif "activities" in prompt.lower():
            return MockGoogleAIResponse("1. Visit temples\n2. Cherry blossom viewing\n3. Traditional tea ceremony\n4. Explore bamboo forest\n5. Visit Nijo Castle")
        elif "accommodations" in prompt.lower():
            return MockGoogleAIResponse("1. Traditional Ryokan\n2. Luxury hotels\n3. Budget-friendly hostels")
        elif "travel" in prompt.lower():
            return MockGoogleAIResponse("Day 1: Arrival and temple visits\nDay 2: Cultural experiences\nDay 3: Nature exploration")
        else:
            return MockGoogleAIResponse("Generic response for: " + prompt)

class TestGoogleAINode(unittest.TestCase):
    """Tests for GoogleAINode class."""
    
    def setUp(self):
        self.model = MockGoogleAIModel()
        self.node = GoogleAINode(
            name="test_node",
            model=self.model,
            prompt_template="Test prompt about {topic}"
        )
    
    def test_initialization(self):
        """Test node initialization."""
        self.assertEqual(self.node.name, "test_node")
        self.assertEqual(self.node.prompt_template, "Test prompt about {topic}")
        
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test node execution."""
        result = await self.node.execute({"topic": "weather"})
        self.assertIn("weather", result.lower())
        
    @pytest.mark.asyncio
    async def test_format_prompt(self):
        """Test prompt formatting."""
        formatted = self.node._format_prompt({"topic": "activities"}, {})
        self.assertEqual(formatted, "Test prompt about activities")
        
        # Test with context
        formatted = self.node._format_prompt({"topic": "basic"}, {"additional": "context"})
        self.assertEqual(formatted, "Test prompt about basic")

class TestGoogleAIIntegration(unittest.TestCase):
    """Tests for GoogleAIIntegration class."""
    
    def setUp(self):
        self.model = MockGoogleAIModel()
        self.integration = GoogleAIIntegration(self.model)
    
    def test_initialization(self):
        """Test integration initialization."""
        self.assertIsNotNone(self.integration.dag)
        self.assertIsNotNone(self.integration.scheduler)
        
    def test_add_node(self):
        """Test adding a node to the integration."""
        node = self.integration.addNode(
            name="weather_info",
            prompt_template="What's the weather like in {location}?",
            dependencies=["location_info"]
        )
        
        self.assertEqual(node.name, "weather_info")
        self.assertEqual(node.prompt_template, "What's the weather like in {location}?")
        self.assertEqual(node.dependencies, ["location_info"])
        
    def test_optimize(self):
        """Test optimization settings."""
        self.integration.optimize({
            "maxParallelCalls": 3,
            "maxExecutionTime": 30000,
            "priorityNodes": ["weather_info"]
        })
        
        # Verify the settings were applied
        self.assertEqual(self.integration.scheduler.max_parallel_nodes, 3)
        self.assertEqual(self.integration.scheduler.max_execution_time, 30000)
        self.assertEqual(self.integration.scheduler.priority_nodes, ["weather_info"])
        
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test execution of the integration DAG."""
        # Add nodes to test
        self.integration.addNode(
            name="weather_info",
            prompt_template="What's the weather like in {location}?",
            dependencies=[]
        )
        
        self.integration.addNode(
            name="activities",
            prompt_template="What activities can I do in {location} with {weather_info}?",
            dependencies=["weather_info"]
        )
        
        # Execute the DAG
        results = await self.integration.execute({"location": "Kyoto"})
        
        # Check that both nodes were executed
        self.assertIn("weather_info", results)
        self.assertIn("activities", results)
        
        # Check result content
        self.assertIn("weather", results["weather_info"].lower())
        self.assertIn("visit", results["activities"].lower())

class TestGoogleAIBatchProcessor(unittest.TestCase):
    """Tests for GoogleAIBatchProcessor class."""
    
    def setUp(self):
        self.model = MockGoogleAIModel()
        self.batch_processor = GoogleAIBatchProcessor(
            model=self.model,
            batchSize=2,
            maxConcurrentBatches=2
        )
    
    def test_initialization(self):
        """Test batch processor initialization."""
        self.assertEqual(self.batch_processor.batch_size, 2)
        self.assertEqual(self.batch_processor.max_concurrent_batches, 2)
        
    @pytest.mark.asyncio
    async def test_process(self):
        """Test batch processing."""
        async def process_item(item, model):
            response = await model.generate_content(f"Info about {item}")
            return {"item": item, "info": response.text()}
        
        items = ["Tokyo", "Paris", "Rome", "New York"]
        
        results = await self.batch_processor.process(items, process_item)
        
        # Check that all items were processed
        self.assertEqual(len(results), len(items))
        
        # Check that each result contains the expected data
        for i, result in enumerate(results):
            self.assertEqual(result["item"], items[i])
            self.assertIn("generic response", result["info"].lower())


if __name__ == "__main__":
    unittest.main()