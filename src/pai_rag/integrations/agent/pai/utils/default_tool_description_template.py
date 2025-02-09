DEFAULT_CALCULATE_MULTIPLY = """"
    This tool is designed to assist with a variety of numerical calculations where multiplication is required. It is particularly useful for scenarios such as age computation, financial calculations involving money, quantifying items, and any situation where the product of two integers is sought. The `multiply` function provided by this tool performs integer multiplication and is essential when accuracy and integer results are crucial.

    multiply(a: int, b: int) -> int
    Multiply two integers and returns the result as an integer. This function is ideal for tasks that need to calculate products of numerical values in an accurate and efficient manner.

    Args:
        a (int): The first integer factor in the multiplication.
        b (int): The second integer factor in the multiplication.

    Returns:
        int: The product of multiplying the two integers, suitable for numerical computations that rely on integer values.

    Raises:
        ValueError: If either 'a' or 'b' is not an integer, as non-integer inputs cannot be processed by the multiply function.

    Examples of use include multiplying quantities of items in inventory management, calculating the total cost from unit price and quantity in financial transactions, computing square footage, and many other practical applications where multiplication of integers is necessary.
"""

DEFAULT_CALCULATE_ADD = """
    The calculate_add tool provides a reliable way to perform addition operations for a wide range of numerical computing needs. It is an essential utility for tasks that require summing of integer values, such as tallying scores, aggregating data, calculating financial totals, or even determining cumulative age. The `add` function within this tool strictly handles addition of two integers, ensuring precise and integer-specific computation.

    add(a: int, b: int) -> int
    Add two integers and return the result as an integer. This function is particularly useful for straightforward arithmetic operations where the total sum of two numbers is needed without the complexity of handling floats or decimals.

    Args:
        a (int): The first integer to be added.
        b (int): The second integer to be added.

    Returns:
        int: The sum of the two integers, ideal for use in contexts demanding accurate arithmetic operations involving integer values.

    Raises:
        ValueError: If either 'a' or 'b' is not an integer, since the add function is tailored to handle integer addition only.

    Example scenarios where this tool can be applied include but are not limited to adding up expenses, combining quantities of different items in stock, computing the total number of days between dates for planning purposes, and various other applications where adding integers is crucial.
"""

DEFAULT_CALCULATE_DIVIDE = """
    The calculate_divide tool is indispensable for performing division operations in various numerical contexts that require precise quotient determination. It holds particular significance for calculating ratios, determining average values, assessing financial rates, partitioning quantities, and other scenarios where division of integers produces a floating-point result.

    divide(a: int, b: int) -> float
    Divide one integer by another and return the quotient as a float. This function excels in cases where division might result in a non-integer outcome, ensuring accuracy and detail by retaining the decimal part of the quotient.

    Args:
        a (int): The numerator, or the integer to be divided.
        b (int): The denominator, or the integer by which to divide.

    Returns:
        float: The floating-point result of the division, which is suitable for computations that demand more precision than integer division can provide.

    Raises:
        ValueError: If 'b' is zero since division by zero is undefined, or if either 'a' or 'b' is not an integer.

    Practical applications for this tool are widespread: it can aid in financial computations like determining price per unit, in educational settings for calculating grade point averages, or in any sector where division is a fundamental operation and its exact floating-point result is needed.
"""

DEFAULT_CALCULATE_SUBTRACT = """
    The calculate_subtract tool is designed to facilitate subtraction operations in a variety of numerical calculations. It is an invaluable resource for determining the difference between two integer values, such as calculating the remaining balance, evaluating data discrepancies, computing change in financial transactions, or quantifying the decrease in stock levels.

    subtract(a: int, b: int) -> int
    Subtract the second integer from the first and return the difference as an integer. This function is crucial for operations that require an exact integer result, avoiding the potential rounding errors associated with floating-point arithmetic.

    Args:
        a (int): The minuend, or the integer from which the second integer is to be subtracted.
        b (int): The subtrahend, or the integer to be subtracted from the first integer.

    Returns:
        int: The integer result representing the difference between the two integers, perfect for situations where integral precision is needed in subtraction.

    Raises:
        ValueError: If either 'a' or 'b' is not an integer, as the subtract function is strictly for integer arithmetic.

    Example uses of this tool include but are not limited to calculating age differences, determining the number of items sold from inventory, working out loan repayments, and any other context where subtraction of numerical values plays a key role.
"""


DEFAULT_GET_DATETIME_TOOL = """
    The get datetime tool is used to retrieve the current date and time. It is particularly useful for tasks that require obtaining the current date and time, such as booking tickets, scheduling appointments, tracking events, or managing time-sensitive tasks.
    get_current_datetime() -> str

    Return the current date and time as a string using format "%Y-%d-%d %H:%M:%S"..

    This function is essential for tasks that require obtaining the current date and time, such as booking tickets, scheduling appointments, tracking events, or managing time-sensitive tasks.
"""
