# Solution in Words: Python OOP Basics

## How to Think About This Problem

### Understanding Classes

Think of a class as a blueprint for creating objects. Like a cookie cutter that makes cookies, a class makes objects.

**Mental Model:**
```
Class = Blueprint
Object = Instance created from blueprint

CookieCutter (class)
  ↓
Cookie (object)
Cookie (object)
Cookie (object)
```

### Step-by-Step Thinking Process

#### 1. Defining a Class

**What we're doing:** Creating a template that describes what objects of this type will have (attributes) and what they can do (methods).

**How to think about it:**
- What properties does this thing have? → Attributes
- What actions can it perform? → Methods
- What information is needed to create one? → Constructor parameters

**Example: BankAccount**
- Properties: balance
- Actions: deposit, withdraw, check balance
- Creation: needs initial balance

#### 2. The __init__ Constructor

**What we're doing:** Setting up a new object with initial values.

**How to think about it:**
- This runs automatically when you create an object
- Use it to set initial attribute values
- `self` refers to the object being created

**Example thought process:**
- "When I create a BankAccount, I need to set the balance" → `self.balance = initial_balance`

#### 3. Instance Methods

**What we're doing:** Defining behaviors that objects can perform.

**How to think about it:**
- Methods always take `self` as first parameter
- `self` lets you access the object's attributes
- Methods can modify or return information about the object

**Example thought process:**
- "A BankAccount can deposit money" → `def deposit(self, amount): self.balance += amount`
- "A BankAccount can withdraw money" → `def withdraw(self, amount): self.balance -= amount`

#### 4. Creating Objects

**What we're doing:** Making instances of a class.

**How to think about it:**
- Call the class name like a function
- Pass arguments that match `__init__` parameters
- Each object is independent

**Example:**
```python
account1 = BankAccount(100)  # Creates account with $100
account2 = BankAccount(50)    # Creates different account with $50
```

### Common Patterns

#### Pattern 1: Getter Methods
Methods that return information about the object.

**Thinking:**
- "I need to know the balance" → `def get_balance(self): return self.balance`

#### Pattern 2: Setter Methods
Methods that modify the object's state.

**Thinking:**
- "I need to change the balance" → `def set_balance(self, new_balance): self.balance = new_balance`

#### Pattern 3: Calculation Methods
Methods that compute values based on attributes.

**Thinking:**
- "I need to calculate distance from origin" → Use attributes to compute result

### BankAccount Example

**Problem:** Create a BankAccount class with deposit and withdraw methods.

**Step-by-step thinking:**

1. **Define the class:**
   ```python
   class BankAccount:
   ```

2. **Create constructor:**
   - Takes initial balance
   - Stores it in `self.balance`

3. **Add deposit method:**
   - Takes amount to deposit
   - Adds to `self.balance`

4. **Add withdraw method:**
   - Takes amount to withdraw
   - Subtracts from `self.balance`
   - Could add validation (check if enough funds)

### Point Class Example

**Problem:** Create a Point class that can calculate distance from origin.

**Thinking:**
- Point has x and y coordinates → `self.x`, `self.y`
- Distance formula: √(x² + y²)
- Need to import math for sqrt

### Key Concepts

1. **self**: Always the first parameter in instance methods
2. **Attributes**: Store data (`self.attribute_name`)
3. **Methods**: Define behavior (`def method_name(self):`)
4. **Encapsulation**: Data and methods together in one object

### Problem-Solving Strategy

1. **Identify the entity:** What are we modeling? (BankAccount, Point, etc.)
2. **List attributes:** What properties does it have?
3. **List behaviors:** What can it do?
4. **Write constructor:** How do we create one?
5. **Write methods:** Implement each behavior
6. **Test:** Create objects and call methods

This OOP approach helps organize code and model real-world concepts effectively.

