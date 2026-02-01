from sympy import symbols, Not, Or, And, Implies, Equivalent
from sympy.logic.boolalg import truth_table

# Khai báo các biến logic
P, Q, R = symbols('P Q R')

# Các biểu thức logic cần tính toán
a = Equivalent(Not(P), Not(Q))
b = Equivalent(Or(P, Not(Q)), P)
c = Not(And(P, Or(Q, R)))
d = Not(And(P, Q))
e = Implies(And(P, Not(P)), Q)
f = Implies(And(P, Q), P)
g = Implies(P, Or(P, Q))

expressions = [a, b, c, d, e, f, g]
labels = ["(¬P) ⇔ (¬Q)", "[P ∨ (¬Q)] ⇔ P", "¬[P ∧ (Q ∨ R)]", "¬(P ∧ Q)", "(P ∧ (¬P)) ⇒ Q", "(P ∧ Q) ⇒ P", "P ⇒ (P ∨ Q)"]

# Xuất bảng chân trị
for expr, label in zip(expressions, labels):
    print(f"Truth Table for: {label}")
    print("=" * 40)
    for val in truth_table(expr, [P, Q, R]):
        print(val)
    print("\n")
