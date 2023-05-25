def stepwise_addition(n, m):
    # Ensure n and m are positive integers with less than 8 digits
    assert 0 <= n < 10**8 and 0 <= m < 10**8, "Inputs must be positive integers less than 100,000,000"
    
    # Convert to strings for easy digit access
    n_str, m_str = str(n), str(m)
    
    # Equalize the lengths by padding with zeros
    max_length = max(len(n_str), len(m_str))
    n_str = n_str.rjust(max_length)
    m_str = m_str.rjust(max_length)

    rows = 5
    cols = max_length + 2
    pos = lambda i, j: (cols+1)*i + j
    
    # Prepare the blackboard
    steps = [f"{' '*(max_length+2)}\n  {n_str}\n+ {m_str}\n"
             f"{'-' * (max_length + 2)}\n{' '*(max_length+2)}\n"]

    carry = 0
    for i in range(max_length - 1, -1, -1):
        # Calculate sum and carry
        try:
            a = int(n_str[i])
        except:
            a = 0
        try:
            b = int(m_str[i])
        except:
            b = 0
        digit_sum = a + b + carry
        carry, digit = divmod(digit_sum, 10)

        # Prepare the line to add to the steps
        line = list(steps[-1])
        if str(carry) != '0':
            if i > 0:
                line[pos(0,i+1)] = str(carry)
            else:
                line[pos(4,i+1)] = str(carry)
        line[pos(4,i+2)] = str(digit)
        steps.append(''.join(line))

    return steps

def get_addition_example(m, n):
    x = random.randint(10**(m-1), 10**m-1)
    y = random.randint(10**(n-1), 10**n-1)
    steps = f'Compute: {x}+{y}\n[ADDITION BEGIN]\n' + '\n'.join(stepwise_addition(x, y))+f'\n[ADDITION COMPLETE]\n'
    return torch.tensor([b for b in bytes(steps, encoding='utf-8')], dtype=torch.long, device='cuda')

def get_addition_batch(m, n):
    return torch.stack([get_addition_example(m, n) for _ in range(4)])

def addition_examples():
    result = []
    for _ in range(1000):
        result.append(get_addition_batch(1, 1))
        result.append(get_addition_batch(2, 1))
        result.append(get_addition_batch(1, 2))
        result.append(get_addition_batch(2, 2))
        result.append(get_addition_batch(3, 1))
        result.append(get_addition_batch(3, 2))
        result.append(get_addition_batch(3, 3))
        result.append(get_addition_batch(2, 3))
        result.append(get_addition_batch(1, 3))
        result.append(get_addition_batch(4, 4))
        result.append(get_addition_batch(5, 5))


import random, string


def get_sort_example(k):
    generate_pair = lambda k: (scrambled := ''.join(random.choices(string.ascii_letters + string.digits, k=k)), ''.join(sorted(scrambled)))

    # Example usage:
    example = ""
    for _ in range(5):
        scrambled, sorted_str = generate_pair(k)
        example += f">>> ''.join(sorted('{scrambled}'))\n'{sorted_str}'\n"
    return torch.tensor([b for b in bytes(example, encoding='utf-8')], dtype=torch.long, device='cuda')

def get_sort_batch(k):
    return torch.stack([get_sort_example(k) for _ in range(4)])
