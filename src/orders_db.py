import json
import logging
from typing import Any, Dict

from langchain.tools import tool

logger = logging.getLogger(__name__)

# Global variable to store orders data
_orders_data: Dict[str, Any] = {}

def load_orders(file_path: str = './data/orders.json') -> Dict[str, Any]:
    """Load orders data from JSON file.
    
    Args:
        file_path: Path to the orders JSON file
        
    Returns:
        Dictionary containing orders data
    """
    global _orders_data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            _orders_data = json.load(f)
        logger.info(f"Orders data loaded successfully from {file_path}")
        return _orders_data
    except FileNotFoundError:
        logger.warning(f"Orders file not found at {file_path}")
        _orders_data = {}
        return {}
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON format in {file_path}")
        _orders_data = {}
        return {}
    except Exception as e:
        logger.error(f"Error loading orders: {e}")
        _orders_data = {}
        return {}

@tool
def lookup_order_tool(order_id: str) -> str:
    """Lookup order status by order ID.

    Use this tool when the customer asks about the status of their order.

    Args:
        order_id: The order ID to lookup
        
    Returns:
        Formatted string with order status information
    """
    if not _orders_data:
        return "База данных заказов не загружена. Обратитесь к администратору."

    if order_id in _orders_data:
        order = _orders_data[order_id]
        status = order['status']

        if status == 'in_transit':
            eta = f"Ожидается через {order['eta_days']} дней. Перевозчик: {order['carrier']}"
            return f"Заказ {order_id} в пути. {eta}"
        elif status == 'delivered':
            return f"Заказ {order_id} доставлен {order['delivered_at']}"
        elif status == 'processing':
            note = order.get('note', '')
            return f"Заказ {order_id} обрабатывается. {note}"
        else:
            return f"Заказ {order_id} имеет статус: {status}"
    else:
        return f"Заказ {order_id} не найден."

def get_orders_data() -> Dict[str, Any]:
    """Get the loaded orders data.
    
    Returns:
        Dictionary containing orders data
    """
    return _orders_data
