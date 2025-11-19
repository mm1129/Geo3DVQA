"""
統一されたエラーハンドリングユーティリティ
Unified Error Handling Utility

このモジュールは、svf_questions_rgb_estimated.py、svf_questions_hard.py、
svf_questions_region_based.pyで使用される統一されたエラーハンドリングを提供します。
"""

from enum import Enum
from typing import Optional, Dict, Any, List
import traceback

try:
    from utils import tqdm_safe_print
except ImportError:
    def tqdm_safe_print(*args, **kwargs):
        print(*args, **kwargs)


class ErrorType(Enum):
    """エラータイプの定義"""
    MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"
    INVALID_INPUT_SIZE = "INVALID_INPUT_SIZE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ErrorHandler:
    """統一されたエラーハンドラー"""
    
    def __init__(self, debug: bool = False):
        """
        Initialize error handler
        
        Args:
            debug: Enable debug mode for detailed error logging
        """
        self.debug = debug
        self.error_log: List[Dict[str, Any]] = []
    
    def handle_error(
        self,
        error_type: ErrorType,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> tuple:
        """
        統一されたエラーハンドリング
        
        Args:
            error_type: Error type enum
            message: Error message
            context: Additional context information
            exception: Exception object if available
            
        Returns:
            tuple: (None, None, None) - Standard return value for errors
        """
        error_info = {
            "type": error_type.value,
            "message": message,
            "context": context or {},
            "exception": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception and self.debug else None
        }
        
        self.error_log.append(error_info)
        
        # Debug mode時は詳細情報を出力
        if self.debug:
            self._log_error(error_info)
        
        return None, None, None
    
    def _log_error(self, error_info: Dict[str, Any]):
        """エラー情報をログ出力"""
        method = error_info["context"].get("method", "unknown")
        error_type = error_info["type"]
        message = error_info["message"]
        
        tqdm_safe_print(f"[{error_type}] {method}: {message}")
        
        # Context情報の出力
        if error_info["context"]:
            for key, value in error_info["context"].items():
                if key != "method":
                    tqdm_safe_print(f"  {key}: {value}")
        
        # Exception情報の出力
        if error_info["exception"]:
            tqdm_safe_print(f"  Exception: {error_info['exception']}")
        
        # Tracebackの出力（debug mode時のみ）
        if error_info.get("traceback") and self.debug:
            tqdm_safe_print(f"  Traceback:\n{error_info['traceback']}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        エラーサマリーを取得
        
        Returns:
            Dict containing error summary statistics
        """
        error_counts = {}
        for error in self.error_log:
            error_type = error["type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "error_counts": error_counts,
            "errors": self.error_log
        }
    
    def clear_log(self):
        """エラーログをクリア"""
        self.error_log.clear()
    
    def has_errors(self) -> bool:
        """エラーが発生したかどうかを確認"""
        return len(self.error_log) > 0

