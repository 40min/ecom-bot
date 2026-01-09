import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.bot import CliBot, StructuredAnswer

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG bot performance on document-based question answering."""
    
    def __init__(
        self,
        bot: CliBot,
        reports_dir: Path,
    ):
        """Initialize the RAG evaluator.
        
        Args:
            bot: The RAG bot instance to evaluate
            reports_dir: Directory to save evaluation reports
        """
        self.bot = bot
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(exist_ok=True)
    
    def _has_valid_citations(self, response: StructuredAnswer) -> bool:
        """Check if response has at least one valid citation.
        
        A valid citation must have non-empty source, page, and snippet fields.
        
        Args:
            response: The bot's structured response
            
        Returns:
            True if at least one valid citation exists
        """
        if not response.citations:
            return False
        
        for citation in response.citations:
            if (
                citation.source
                and citation.page is not None
                and citation.snippet
                and len(citation.source.strip()) > 0
                and len(citation.snippet.strip()) > 0
            ):
                return True
        
        return False
    
    def _uses_fallback_response(self, response: StructuredAnswer) -> bool:
        """Check if the response indicates the bot doesn't know the answer.
        
        This checks if the answer uses the fallback phrase indicating
        the information is not available in the knowledge base.
        
        Args:
            response: The bot's structured response
            
        Returns:
            True if response indicates lack of knowledge
        """
        # Get the fallback response from the bot's person configuration
        fallback_phrase = self.bot.person.no_info_fallback_response.lower()
        answer_text = response.answer.lower()
        
        # 1. Check for exact fallback phrase (original logic)
        if fallback_phrase in answer_text:
            return True
            
        # 2. Check for partial matches of the fallback phrase
        # Split fallback into significant words (length > 3) to avoid common prepositions
        fallback_words = [w for w in re.findall(r'\w+', fallback_phrase) if len(w) > 3]
        if fallback_words:
            # If at least 50% of significant words from fallback are present
            matches = sum(1 for word in fallback_words if word in answer_text)
            if matches / len(fallback_words) >= 0.5:
                return True

        # 3. Check for common lack-of-knowledge indicators
        fallback_indicators = [
            "не знаю",
            "не ведаю",
            "не приходилось слыхать",
            "нет информации",
            "не могу ответить",
            "нет данных",
            "информация отсутствует",
            "не в курсе",
        ]
        
        return any(indicator in answer_text for indicator in fallback_indicators)
    
    def _evaluate_in_scope(self, response: StructuredAnswer) -> dict:
        """Evaluate an in-scope question (oos=false).
        
        PASS if: response has at least 1 citation with valid source, page, and snippet
        FAIL if: no citations or citations are empty/invalid
        
        Args:
            response: The bot's structured response
            
        Returns:
            Evaluation result dictionary
        """
        has_valid_citations = self._has_valid_citations(response)
        
        return {
            "pass": has_valid_citations,
            "has_citations": has_valid_citations,
            "reason": "Has valid citations" if has_valid_citations else "No valid citations",
        }
    
    def _evaluate_out_of_scope(self, response: StructuredAnswer) -> dict:
        """Evaluate an out-of-scope question (oos=true).
        
        PASS if: answer uses fallback phrase AND no citations provided
        FAIL if: bot hallucinated an answer or provided fake citations
        
        Args:
            response: The bot's structured response
            
        Returns:
            Evaluation result dictionary
        """
        uses_fallback = self._uses_fallback_response(response)
        has_citations = self._has_valid_citations(response)
        
        # PASS only if both conditions are met
        passed = uses_fallback and not has_citations
        
        if not passed:
            if has_citations:
                reason = "Provided citations for out-of-scope question (hallucination)"
            else:
                reason = "Did not use fallback response"
        else:
            reason = "Correctly used fallback and no citations"
        
        return {
            "pass": passed,
            "has_citations": has_citations,
            "uses_fallback": uses_fallback,
            "reason": reason,
        }
    
    def evaluate_single(self, question: str, oos: bool, category: str) -> dict:
        """Evaluate a single question.
        
        Args:
            question: The test question
            oos: Whether the question is out-of-scope
            category: Question category for grouping
            
        Returns:
            Evaluation result dictionary
        """
        try:
            # Get bot response
            session_id = self.bot.get_new_session_id(user_id="rag_eval")
            response, _ = self.bot.ask(question, session_id)
            
            # Evaluate based on oos flag
            if oos:
                eval_result = self._evaluate_out_of_scope(response)
            else:
                eval_result = self._evaluate_in_scope(response)
            
            return {
                "q": question,
                "pass": eval_result["pass"],
                "oos": oos,
                "category": category,
                "answer": response.answer,
                "has_citations": eval_result["has_citations"],
                "confidence": response.confidence,
                "citations_count": len(response.citations),
                "reason": eval_result.get("reason", ""),
                "error": None,
            }
            
        except Exception as e:
            logger.error(f"Error evaluating question '{question}': {e}")
            return {
                "q": question,
                "pass": False,
                "oos": oos,
                "category": category,
                "answer": None,
                "has_citations": False,
                "confidence": None,
                "citations_count": 0,
                "reason": f"Evaluation error: {str(e)}",
                "error": str(e),
            }
    
    def evaluate(self, prompts_file: Path) -> dict:
        """Run full evaluation on all test questions.
        
        Args:
            prompts_file: Path to JSON file with test questions
            
        Returns:
            Complete evaluation report dictionary
        """
        # Load test questions
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_questions = data.get("prompts", [])
        except FileNotFoundError:
            logger.error(f"Test questions file not found: {prompts_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in test questions file: {e}")
            raise
        
        if not test_questions:
            logger.warning("No test questions found in file")
            return {
                "pass_rate": 0.0,
                "total_count": 0,
                "passed_count": 0,
                "failed_count": 0,
                "items": [],
            }
        
        # Evaluate each question
        items = []
        for item in test_questions:
            question = item.get("question", "")
            oos = item.get("oos", False)
            category = item.get("category", "unknown")
            
            result = self.evaluate_single(question, oos, category)
            items.append(result)
        
        # Calculate statistics
        total_count = len(items)
        passed_count = sum(1 for item in items if item["pass"])
        failed_count = total_count - passed_count
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0.0
        
        # Build report
        report = {
            "pass_rate": round(pass_rate, 2),
            "total_count": total_count,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "target_pass_rate": 80.0,
            "meets_target": pass_rate >= 80.0,
            "items": items,
        }
        
        # Save to file
        output_path = self.reports_dir / "rag_eval.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        return report
