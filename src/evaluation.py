"""RAGAS evaluation module for RAG pipeline quality assessment."""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)

from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrock

from .retrieval import RAGPipeline
from .llm_providers import OllamaProvider, BedrockProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG pipeline using RAGAS metrics."""

    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the evaluator.

        Args:
            rag_pipeline: RAG pipeline instance to evaluate
        """
        self.rag_pipeline = rag_pipeline

    def prepare_eval_dataset(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None
    ) -> Dataset:
        """
        Prepare evaluation dataset in RAGAS format.

        Args:
            questions: List of questions
            ground_truths: List of ground truth answers (optional)
            answers: List of generated answers (optional, will generate if not provided)
            contexts: List of context lists (optional, will retrieve if not provided)

        Returns:
            Dataset object for RAGAS evaluation
        """
        if answers is None or contexts is None:
            logger.info("Generating answers and retrieving contexts...")
            results = self.rag_pipeline.batch_query(questions)
            if answers is None:
                answers = [r["answer"] for r in results]
            if contexts is None:
                contexts = [r["contexts"] for r in results]

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths is not None:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)
        logger.info(f"Prepared evaluation dataset with {len(questions)} samples")
        return dataset

    def evaluate(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None,
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline using RAGAS metrics.

        Args:
            questions: List of questions
            ground_truths: List of ground truth answers (required for some metrics)
            answers: List of generated answers (will generate if not provided)
            contexts: List of context lists (will retrieve if not provided)
            metrics: List of RAGAS metrics to use (default: all available metrics)

        Returns:
            Dictionary with evaluation results
        """
        # Prepare dataset
        dataset = self.prepare_eval_dataset(questions, ground_truths, answers, contexts)

        # Select metrics based on available data
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
            ]
            # Add metrics that require ground truth only if available
            if ground_truths is not None:
                metrics.extend([
                    context_recall,
                    # answer_correctness, # Disabled for smaller models that struggle with JSON output
                ])

        # Create a compatible LangChain chat model for ragas based on provider type
        llm_provider = self.rag_pipeline.llm_provider

        if isinstance(llm_provider, OllamaProvider):
            chat_model = ChatOllama(
                model=llm_provider.model_name,
                base_url=llm_provider.base_url
            )
        elif isinstance(llm_provider, BedrockProvider):
            # Bedrock with Claude models (native ChatBedrock support)
            chat_model = ChatBedrock(
                model_id=llm_provider.model_name,
                region_name=llm_provider.region_name,
                client=llm_provider.client
            )
        else:
            raise ValueError(f"Unsupported LLM provider type: {type(llm_provider)}")

        # Run evaluation
        try:
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=chat_model,
                embeddings=self.rag_pipeline.embedding_provider
            )
            logger.info("Evaluation completed successfully")
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def evaluate_from_file(
        self,
        csv_path: str,
        question_col: str = "question",
        ground_truth_col: Optional[str] = "ground_truth",
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline from a CSV file.

        Args:
            csv_path: Path to CSV file with questions and ground truths
            question_col: Name of question column
            ground_truth_col: Name of ground truth column (optional)
            metrics: List of RAGAS metrics to use

        Returns:
            Dictionary with evaluation results
        """
        df = pd.read_csv(csv_path)
        questions = df[question_col].tolist()

        ground_truths = None
        if ground_truth_col and ground_truth_col in df.columns:
            ground_truths = df[ground_truth_col].tolist()

        return self.evaluate(questions, ground_truths=ground_truths, metrics=metrics)

    def compare_configurations(
        self,
        questions: List[str],
        ground_truths: List[str],
        top_k_values: List[int] = [3, 5, 10]
    ) -> pd.DataFrame:
        """
        Compare RAG performance with different configurations.

        Args:
            questions: List of questions
            ground_truths: List of ground truth answers
            top_k_values: List of top_k values to test

        Returns:
            DataFrame with comparison results
        """
        results = []

        for top_k in top_k_values:
            logger.info(f"Evaluating with top_k={top_k}")

            # Generate answers with specific top_k
            rag_results = []
            for question in questions:
                result = self.rag_pipeline.query(question, top_k=top_k)
                rag_results.append(result)

            answers = [r["answer"] for r in rag_results]
            contexts = [r["contexts"] for r in rag_results]

            # Evaluate
            eval_result = self.evaluate(
                questions=questions,
                ground_truths=ground_truths,
                answers=answers,
                contexts=contexts
            )

            # Store results
            result_dict = {"top_k": top_k}
            result_dict.update(eval_result)
            results.append(result_dict)

        df = pd.DataFrame(results)
        logger.info("Configuration comparison completed")
        return df

    def generate_report(
        self,
        eval_result: Any,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            eval_result: Evaluation result (can be RAGAS EvaluationResult or dict)
            output_path: Path to save report (optional)

        Returns:
            Report string
        """
        report = "=" * 50 + "\n"
        report += "RAG Pipeline Evaluation Report\n"
        report += "=" * 50 + "\n\n"

        # Convert RAGAS EvaluationResult to dict if needed
        if hasattr(eval_result, 'to_pandas'):
            # RAGAS returns an EvaluationResult object
            # Convert to pandas and get mean scores
            df = eval_result.to_pandas()
            result_dict = {}
            # Get mean of each metric column (skip non-numeric columns)
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    result_dict[col] = df[col].mean()
        elif isinstance(eval_result, dict):
            result_dict = eval_result
        else:
            # Try to convert to dict
            result_dict = dict(eval_result)

        for metric, value in result_dict.items():
            if isinstance(value, (int, float)):
                report += f"{metric}: {value:.4f}\n"

        report += "\n" + "=" * 50 + "\n"

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report
