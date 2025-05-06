
from enum import Enum

WHITE_SPACE = " \t\n\r"
OPERATORS = "+-*/="
OPERATORS_PRECEDENCE = {
    "=": 50,
    "+": 150,
    "-": 150,
    "*": 200,
    "/": 200,
}


class Panic:
    @staticmethod
    def call(type: str, message: str):
        print(f'\n{type} Panic: {message}\n')
        exit(0)


def scanner_panic(message): return Panic.call('Scanner', message)
def parser_panic(message): return Panic.call('Parser', message)
def evaluator_panic(message): return Panic.call('Evaluator', message)


class TokenType(Enum):
    NUMBER = 1
    OPERATOR = 2
    EOF = 3
    ECHO = 4
    ASSIGN = 5
    IDENTIFIER = 6


class Token:
    def __init__(self, token_type: TokenType, value: str) -> None:
        self.token_type = token_type
        self.value = value

    def __repr__(self) -> str:
        return f"{self.token_type}({self.value})"


class Scanner:
    def __init__(self, input: str) -> None:
        self.input = input
        self.position = 0
        self.current_char = self.input[self.position]
        self.tokens: list[Token] = []

    def advance(self) -> bool:
        self.position += 1
        if self.position >= len(self.input):
            return False

        self.current_char = self.input[self.position]
        return True

    def scan(self) -> list[Token]:
        while self.position < len(self.input):
            if self.current_char in WHITE_SPACE:
                self.advance()
                continue

            last_token = self.try_get_next_token()
            self.tokens.append(last_token)

        self.tokens.append(Token(TokenType.EOF, ""))
        return self.tokens

    def try_get_next_token(self) -> Token:
        if token := (
                self.scan_number() or
                self.scan_operator() or
                self.scan_identifier()):
            return token

        scanner_panic(f"Unexpected character: {self.current_char}")

    def scan_number(self) -> Token:
        if not self.current_char.isdigit():
            return None

        start_pos = self.position

        while self.current_char.isdigit():
            if not self.advance():  # TODO
                break

        return Token(TokenType.NUMBER, self.input[start_pos:self.position])

    def scan_operator(self) -> Token:
        if self.current_char not in "+-*/=":
            return None
        if self.current_char == "=":
            self.advance()
            return Token(TokenType.ASSIGN)

        token = Token(TokenType.OPERATOR, self.current_char)
        self.advance()
        return token

    def scan_identifier(self) -> Token:
        if not self.current_char.isalpha():
            return None

        start_pos = self.position

        while self.current_char.isalnum():
            if not self.advance():
                break

        result = self.input[start_pos:self.position]
        if result == "echo":
            return Token(TokenType.ECHO, '')

        return Token(TokenType.IDENTIFIER, result)


class ASTNode:
    def __init__(self, token: Token, ast: list = None) -> None:
        self.token = token
        self.children: list[ASTNode] = ast if ast else []

    def __repr__(self):
        return self._repr()

    def _repr(self, depth=0):
        tabs = '\t' * depth
        result = f"{tabs}(Node) {self.token}\n"
        if self.children:
            result += f"{tabs}{{\n"
            result += ''.join(child._repr(depth + 1)
                              for child in self.children)
            result += f"{tabs}}}\n"
        return result


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.position = 0
        self.current = self.tokens[self.position]

    def advance(self) -> bool:
        self.position += 1
        if self.position < len(self.tokens):
            self.current = self.tokens[self.position]
            return True
        return False

    def parse(self) -> ASTNode:
        node = ASTNode(Token(TokenType.EOF, ""))

        while True:
            if self.position >= len(self.tokens):
                parser_panic("Unexpected end of input")
                break

            if self.current.token_type == TokenType.EOF:
                break

            if (node := self.parse_math_expr() or self.parse_echo()) is None:
                parser_panic(f"Unexpected token: {self.current}")

        return node

    def parse_echo(self) -> ASTNode:
        if self.current.token_type != TokenType.ECHO:
            return None

        node = ASTNode(self.current)
        self.advance()
        node.children.append(self.parse())

        return node

    def parse_identifier(self) -> ASTNode:
        if self.current.token_type != TokenType.IDENTIFIER:
            return None

        id = self.current
        self.advance()
        if self.current.token_type == TokenType.ASSIGN:
            node = ASTNode(self.current, [ASTNode(id)])
            self.advance()
            node.children.append(self.parse_math_expr())
            return node
        else:
            return ASTNode(id)

    def parse_number(self) -> ASTNode:
        if self.current.token_type != TokenType.NUMBER:
            return None

        node = ASTNode(self.current)
        self.advance()
        return node

    def parse_number_or_identifier(self) -> ASTNode:
        return self.parse_number() or self.parse_identifier()

    def parse_math_expr(self) -> ASTNode:
        return self.parse_expression(0)

    def parse_expression(self, min_precedence: int) -> ASTNode:
        left = self.parse_number_or_identifier()

        if left is None:
            return None

        while self.current.token_type == TokenType.OPERATOR and \
                OPERATORS_PRECEDENCE[self.current.value] >= min_precedence:
            operator = self.current
            self.advance()

            next_min_precedence = OPERATORS_PRECEDENCE[operator.value] + 1
            left = ASTNode(
                operator, [left, self.parse_expression(next_min_precedence)])

        return left


class Evaluator:
    def __init__(self, ast: ASTNode) -> None:
        self.ast = ast
        self.position = 0
        self.current_number = 0
        self.variables = {
            'myVariable1': 1000000,
        }

    def evaluate(self) -> int:
        self.current_number = self._evaluate(self.ast)
        print(f"Result: {self.current_number}")

    def _evaluate(self, node: ASTNode) -> int:
        match node.token.token_type:
            case TokenType.ASSIGN:
                return self._evaluate_assign(node)
            case TokenType.NUMBER:
                return int(node.token.value)
            case TokenType.IDENTIFIER:
                return self._evaluate_identifier(node)
            case TokenType.ECHO:
                return self._evaluate_echo(node)
            case _:
                return self._evaluate_operator(node)

    def _evaluate_assign(self, node: ASTNode) -> int:
        left = node.children[0]
        right = node.children[1]

        if left.token.token_type != TokenType.IDENTIFIER:
            evaluator_panic(
                f"Expected identifier, got {left.token.token_type}")

        value = self._evaluate(right)
        self.variables[left.token.value] = value
        return value

    def _evaluate_identifier(self, node: ASTNode) -> int:
        if node.token.value in self.variables:
            return self.variables[node.token.value]
        evaluator_panic(f"Undefined variable: {node.token.value}")

    def _evaluate_echo(self, node: ASTNode) -> int:
        value = self._evaluate(node.children[0])
        print(value)
        return value

    def _evaluate_operator(self, node: ASTNode) -> int:
        left = self._evaluate(node.children[0])
        right = self._evaluate(node.children[1])

        match node.token.value:
            case "+": return left + right
            case "-": return left - right
            case "*": return left * right
            case "/": return left // right
            case _: parser_panic(f"Unknown operator: {node.token.value}")


# EGS 1: echo 2 + 2 * 2
# EGS 2: y = myVariable1 - 4 / 2
# EGS 3: echo x = 34 + myVariable1 + 42 * 12 / 002

lexer = Scanner("echo x = 34 + myVariable1 + 42 * 12 / 002")
tokens = lexer.scan()

parser = Parser(tokens)
ast = parser.parse()
print(ast)
evaluator = Evaluator(ast)
evaluator.evaluate()
