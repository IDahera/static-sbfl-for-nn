import random


def generate_rand_dense_seq(layers: int, layer_size_range: tuple[int, int]) -> list[int]:
    """
    Generate a list of random dense layer sizes.
    
    Args:
        layers (int): Number of layers to generate
        layer_size_range (tuple): Tuple containing (min_size, max_size)
    
    Returns:
        list: List of random layer sizes
    """
    randomized_list = [random.randint(layer_size_range[0], layer_size_range[1]) for _ in range(layers)]
    randomized_list.sort(reverse=True)  # Sort in descending order
    
    return randomized_list


def generate_rand_combs(num_combinations: int, ranges: list[tuple[int, int]]):
    """
    Generate random combinations of values within specified ranges.
    
    Args:
        num_combinations (int): Number of random combinations to generate
        ranges (list): List of tuples, where each tuple contains (min_value, max_value)
    
    Returns:
        list: List of tuples containing random value combinations
    """
    combinations = []
    
    for _ in range(num_combinations):  # Using the built-in range function
        combination = []
        for min_val, max_val in ranges:
            # Generate a random integer between min_val and max_val (inclusive)
            value = random.randint(min_val, max_val)
            
            combination.append(value)
         
        if tuple(combination) not in combinations:
            combinations.append(tuple(combination))
    
    return combinations
    