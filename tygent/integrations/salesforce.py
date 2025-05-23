"""
Salesforce Integration for Tygent

This module provides optimized integration with Salesforce and Einstein AI services.
"""
from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
from ..dag import DAG
from ..nodes import BaseNode, LLMNode
from ..scheduler import Scheduler

class SalesforceNode(BaseNode):
    """A node that interacts with Salesforce and Einstein AI services."""
    
    def __init__(
        self, 
        name: str,
        connection: Any,
        operation_type: str,
        sobject: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize a Salesforce node.
        
        Args:
            name: The name of the node
            connection: Salesforce connection instance
            operation_type: Type of operation ("query", "create", "update", "delete", "einstein")
            sobject: Salesforce object type for the operation
            dependencies: List of node names this node depends on
        """
        super().__init__(name, dependencies=dependencies or [])
        self.connection = connection
        self.operation_type = operation_type
        self.sobject = sobject
        self.kwargs = kwargs
    
    async def execute(self, inputs: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """
        Execute the node by performing a Salesforce operation.
        
        Args:
            inputs: Input values for the node
            context: Execution context
            
        Returns:
            The response from the Salesforce operation
        """
        context = context or {}
        
        # Execute the appropriate Salesforce operation
        try:
            if self.operation_type == "query":
                return await self._execute_query(inputs, context)
            elif self.operation_type == "create":
                return await self._execute_create(inputs, context)
            elif self.operation_type == "update":
                return await self._execute_update(inputs, context)
            elif self.operation_type == "delete":
                return await self._execute_delete(inputs, context)
            elif self.operation_type == "einstein":
                return await self._execute_einstein(inputs, context)
            else:
                raise ValueError(f"Unknown operation type: {self.operation_type}")
        except Exception as e:
            # Add robust error handling
            print(f"Error executing Salesforce node {self.name}: {e}")
            raise
    
    async def _execute_query(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a SOQL query."""
        query = inputs.get("query", "")
        if not query and "soql" in self.kwargs:
            query = self.kwargs["soql"]
        
        # Handle different Salesforce client libraries
        if hasattr(self.connection, "query_async"):
            # Async query
            return await self.connection.query_async(query)
        elif hasattr(self.connection, "query"):
            # Sync query in async wrapper
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.connection.query, query)
        elif self.sobject and hasattr(self.connection, "sobject"):
            # Using the sobject interface
            sobject = self.connection.sobject(self.sobject)
            if "id" in inputs:
                # Retrieve by ID
                if hasattr(sobject, "retrieve_async"):
                    return await sobject.retrieve_async(inputs["id"])
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, sobject.retrieve, inputs["id"])
            else:
                # Query with conditions
                conditions = inputs.get("conditions", {})
                if hasattr(sobject, "find_async"):
                    return await sobject.find_async(conditions)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, sobject.find, conditions)
    
    async def _execute_create(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Create a Salesforce record."""
        if not self.sobject:
            raise ValueError("sobject must be specified for create operations")
        
        data = inputs.get("data", {})
        sobject = self.connection.sobject(self.sobject)
        
        if hasattr(sobject, "create_async"):
            return await sobject.create_async(data)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, sobject.create, data)
    
    async def _execute_update(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Update a Salesforce record."""
        if not self.sobject:
            raise ValueError("sobject must be specified for update operations")
        
        record_id = inputs.get("id")
        if not record_id:
            raise ValueError("id must be provided for update operations")
        
        data = inputs.get("data", {})
        data["Id"] = record_id
        sobject = self.connection.sobject(self.sobject)
        
        if hasattr(sobject, "update_async"):
            return await sobject.update_async(data)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, sobject.update, data)
    
    async def _execute_delete(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Delete a Salesforce record."""
        if not self.sobject:
            raise ValueError("sobject must be specified for delete operations")
        
        record_id = inputs.get("id")
        if not record_id:
            raise ValueError("id must be provided for delete operations")
        
        sobject = self.connection.sobject(self.sobject)
        
        if hasattr(sobject, "delete_async"):
            return await sobject.delete_async(record_id)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, sobject.delete, record_id)
    
    async def _execute_einstein(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute an Einstein AI API call."""
        # Einstein API calls are typically done via REST
        import aiohttp
        
        endpoint = inputs.get("endpoint", self.kwargs.get("endpoint"))
        if not endpoint:
            raise ValueError("endpoint must be provided for Einstein operations")
        
        # Build the full URL
        if not endpoint.startswith("http"):
            # Relative endpoint, prepend instance URL
            instance_url = getattr(self.connection, "instance_url", "")
            endpoint = f"{instance_url}/services/apexrest/einstein/{endpoint}"
        
        # Get authentication token
        access_token = getattr(self.connection, "access_token", "")
        if not access_token:
            # Try to get it from the session
            access_token = getattr(getattr(self.connection, "session", {}), "access_token", "")
        
        if not access_token:
            raise ValueError("Cannot get access token from connection")
        
        # Make the API call
        data = inputs.get("data", {})
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=data, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise ValueError(f"Einstein API error ({response.status}): {error_text}")
                return await response.json()


class SalesforceIntegration:
    """Integration with Salesforce and Einstein AI for optimized execution."""
    
    def __init__(self, connection: Any, **kwargs):
        """
        Initialize the Salesforce integration.
        
        Args:
            connection: Salesforce connection instance
            **kwargs: Additional configuration options
        """
        self.connection = connection
        self.config = kwargs
        self.dag = DAG()
        self.scheduler = Scheduler()
    
    def create_query_node(
        self, 
        name: str, 
        sobject: Optional[str] = None, 
        soql: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> SalesforceNode:
        """
        Create a Salesforce query node.
        
        Args:
            name: The name of the node
            sobject: Salesforce object type
            soql: SOQL query string
            dependencies: List of node names this node depends on
            
        Returns:
            The created Salesforce node
        """
        node = SalesforceNode(
            name=name,
            connection=self.connection,
            operation_type="query",
            sobject=sobject,
            dependencies=dependencies,
            soql=soql,
            **kwargs
        )
        self.dag.add_node(node)
        return node
    
    def create_crud_node(
        self, 
        name: str, 
        operation: str,
        sobject: str,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> SalesforceNode:
        """
        Create a Salesforce CRUD operation node.
        
        Args:
            name: The name of the node
            operation: Operation type ("create", "update", "delete")
            sobject: Salesforce object type
            dependencies: List of node names this node depends on
            
        Returns:
            The created Salesforce node
        """
        if operation not in ["create", "update", "delete"]:
            raise ValueError(f"Invalid CRUD operation: {operation}")
        
        node = SalesforceNode(
            name=name,
            connection=self.connection,
            operation_type=operation,
            sobject=sobject,
            dependencies=dependencies,
            **kwargs
        )
        self.dag.add_node(node)
        return node
    
    def create_einstein_node(
        self, 
        name: str, 
        endpoint: str,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> SalesforceNode:
        """
        Create an Einstein AI API node.
        
        Args:
            name: The name of the node
            endpoint: Einstein API endpoint
            dependencies: List of node names this node depends on
            
        Returns:
            The created Salesforce node
        """
        node = SalesforceNode(
            name=name,
            connection=self.connection,
            operation_type="einstein",
            dependencies=dependencies,
            endpoint=endpoint,
            **kwargs
        )
        self.dag.add_node(node)
        return node
    
    def optimize(self, constraints: Optional[Dict[str, Any]] = None) -> 'SalesforceIntegration':
        """
        Apply optimization settings to the DAG.
        
        Args:
            constraints: Resource constraints to apply
            
        Returns:
            Self for chaining
        """
        constraints = constraints or {}
        
        # Configure the scheduler
        max_parallel = constraints.get("max_concurrent_calls", 4)
        max_time = constraints.get("max_execution_time", 60000)
        priority_nodes = constraints.get("priority_nodes", [])
        
        # Apply configuration to scheduler
        self.scheduler.max_parallel_nodes = max_parallel
        self.scheduler.max_execution_time = max_time
        self.scheduler.priority_nodes = priority_nodes
        
        return self
    
    async def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the DAG with Salesforce nodes.
        
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


class TygentBatchProcessor:
    """Process batches of Salesforce operations efficiently."""
    
    def __init__(
        self, 
        connection: Any, 
        batch_size: int = 200, 
        concurrent_batches: int = 3,
        error_handling: str = "continue"
    ):
        """
        Initialize the batch processor.
        
        Args:
            connection: Salesforce connection
            batch_size: Maximum size of each batch
            concurrent_batches: Maximum number of concurrent batch executions
            error_handling: How to handle errors ("continue" or "abort")
        """
        self.connection = connection
        self.batch_size = batch_size
        self.concurrent_batches = concurrent_batches
        self.error_handling = error_handling
        self.semaphore = asyncio.Semaphore(concurrent_batches)
    
    async def query(self, soql: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute a large SOQL query in batches.
        
        Args:
            soql: SOQL query
            **kwargs: Additional query options
            
        Returns:
            List of query results
        """
        # Check if we need to use the Bulk API for very large datasets
        use_bulk = kwargs.get("use_bulk", False)
        
        if use_bulk and hasattr(self.connection, "bulk"):
            # Use Bulk API
            bulk = self.connection.bulk(self.connection.session_id)
            results = []
            
            # Execute the query
            job = bulk.create_query_job(kwargs.get("sobject", ""), contentType="JSON")
            batch = bulk.query(job, soql)
            bulk.wait_for_batch(job, batch)
            
            # Get results
            for result in bulk.get_all_results_for_query_batch(batch, job):
                results.extend(result)
            
            bulk.close_job(job)
            return results
        else:
            # Use standard query with auto-paging
            if hasattr(self.connection, "query_all_async"):
                return await self.connection.query_all_async(soql)
            elif hasattr(self.connection, "query_all"):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.connection.query_all, soql)
            else:
                # Manual implementation with query/query_more
                result = await self._execute_query(soql)
                all_records = result.get("records", [])
                
                # Handle pagination
                while not result.get("done", True):
                    next_records_url = result.get("nextRecordsUrl")
                    if not next_records_url:
                        break
                        
                    if hasattr(self.connection, "query_more_async"):
                        result = await self.connection.query_more_async(next_records_url)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, self.connection.query_more, next_records_url
                        )
                    
                    all_records.extend(result.get("records", []))
                
                return all_records
    
    async def _execute_query(self, soql: str) -> Dict[str, Any]:
        """Execute a single SOQL query."""
        if hasattr(self.connection, "query_async"):
            return await self.connection.query_async(soql)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.connection.query, soql)
    
    async def bulk_operation(
        self,
        records: List[Dict[str, Any]],
        operation: Callable[[List[Dict[str, Any]]], Any],
        **kwargs
    ) -> List[Any]:
        """
        Process a large set of records in optimized batches.
        
        Args:
            records: List of records to process
            operation: Function to process each batch
            **kwargs: Additional operation options
            
        Returns:
            Results of batch operations
        """
        # Split records into batches
        batches = [
            records[i:i + self.batch_size] 
            for i in range(0, len(records), self.batch_size)
        ]
        results = []
        errors = []
        
        async def process_batch(batch_records):
            async with self.semaphore:
                try:
                    batch_result = await operation(batch_records)
                    return {"success": True, "result": batch_result}
                except Exception as e:
                    return {"success": False, "error": str(e)}
        
        # Process batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Process results
        for batch_result in batch_results:
            if batch_result["success"]:
                results.append(batch_result["result"])
            else:
                errors.append(batch_result["error"])
                if self.error_handling == "abort":
                    raise ValueError(f"Batch operation failed: {batch_result['error']}")
        
        return {"results": results, "errors": errors}