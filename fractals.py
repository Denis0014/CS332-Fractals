import re
import tkinter
import numpy as np
from typing import Union, Any, Sequence

WIDTH = 300
HEIGHT = 300

Shape = Union["Point", "Line"]
Number = Union[int, float]


class BasePoint:
    def __init__(self, x: Number, y: Number) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        features = ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items())
        if features:
            return f"{self.__class__.__name__}({self.x}, {self.y}, {features})"
        return f"{self.__class__.__name__}({self.x}, {self.y})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __hash__(self) -> int:
        return hash((self.x, self.y))


class Point(BasePoint):
    def __init__(self, x: Number, y: Number) -> None:
        super().__init__(x, y)
        self.__features = dict[str, Any]()

    def __getitem__(self, item: str) -> Any:
        return self.__features.get(item, None)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.__features[key] = value

    def __repr__(self) -> str:
        features = ", ".join(f"{k}={repr(v)}" for k, v in self.__features.items())
        if features:
            return f"{self.__class__.__name__}({self.x}, {self.y}, {features})"
        return f"{self.__class__.__name__}({self.x}, {self.y})"


class BaseLine:
    def __init__(self, start: BasePoint, end: BasePoint) -> None:
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start}, {self.end})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseLine):
            return False
        return self.start == other.start and self.end == other.end

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __hash__(self) -> int:
        return hash((self.start, self.end))


class Line(BaseLine):
    def __init__(self, start: BasePoint, end: BasePoint) -> None:
        super().__init__(start, end)
        self.__features = dict[str, Any]()

    def __getitem__(self, item: str) -> Any:
        return self.__features.get(item, None)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.__features[key] = value

    def __repr__(self) -> str:
        features = ", ".join(f"{k}={repr(v)}" for k, v in self.__features.items())
        if features:
            return f"{self.__class__.__name__}({self.start}, {self.end}, {features})"
        return f"{self.__class__.__name__}({self.start}, {self.end})"


class BaseCanvas:
    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive values, got " + str(width) + " and " + str(height))
        
        self.width = width
        self.height = height
        self.grid_matrix: np.ndarray = np.zeros((height, width), dtype=bool)
        self.points: list[Point] = []
        self.lines: list[Line] = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.width}, {self.height}, points_count={len(self.points)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseCanvas):
            return False
        return (
            self.width == other.width
            and self.height == other.height
            and self.grid_matrix == other.grid_matrix
        )

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __add__(self, shape: Shape) -> "BaseCanvas":
        if isinstance(shape, Point):
            if 0 <= shape.x < self.width - 1 and 0 <= shape.y < self.height - 1:
                self.grid_matrix[int(shape.y), int(shape.x)] = True
            self.points.append(shape)

            return self

        else:
            self.lines.append(shape)

        return self

    def __sub__(self, point: Point) -> "BaseCanvas":
        if not (0 <= point.x < self.width - 1 and 0 <= point.y < self.height - 1):
            raise ValueError("Point coordinates must be within the canvas dimensions, got " + str(point.x) + ", " + str(point.y))

        self.grid_matrix[int(point.y), int(point.x)] = False
        self.points.remove(point)
        return self

    def __contains__(self, point: Point) -> bool:
        if not (0 <= point.x < self.width - 1 and 0 <= point.y < self.height - 1):
            return False

        return self.grid_matrix[int(point.y), int(point.x)]

    def __next__(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.grid_matrix[y, x]:
                    return Point(x, y)
        raise StopIteration

    def __iter__(self):
        return self

    def clear(self) -> None:
        self.grid_matrix = np.zeros((self.height, self.width), dtype=bool)
        self.points.clear()


class TkinterCanvas(BaseCanvas):
    def __init__(self, width: int = WIDTH, height: int = HEIGHT) -> None:
        super().__init__(width, height)
        self.tk_canvas = tkinter.Canvas(
            width=self.width, height=self.height, bg="white"
        )
        self.tk_canvas.pack()

    def make_points(self, points: Sequence[Point]) -> None:
        for point in points:
            self += point

    def make_lines(self, lines: Sequence[Line]) -> None:
        for line in lines:
            if line.start == line.end:
                continue

            self += line

    def transform(self, matrix: np.ndarray) -> None:
        if matrix.shape != (3, 3):
            raise ValueError("Transformation matrix must be 3x3.")
        if np.linalg.det(matrix) == 0:
            raise ValueError("Transformation matrix must be invertible (non-singular).")

        self.points.clear()
        self.grid_matrix = np.zeros((self.height, self.width), dtype=bool)

        for i in range(len(self.lines)):
            line = self.lines[i]
            start_vec = np.array([line.start.x, line.start.y, 1])
            end_vec = np.array([line.end.x, line.end.y, 1])
            new_start_vec = matrix @ start_vec
            new_end_vec = matrix @ end_vec

            x0, y0 = int(new_start_vec[0]), int(new_start_vec[1])
            x1, y1 = int(new_end_vec[0]), int(new_end_vec[1])

            point1 = Point(x0, y0)
            point2 = Point(x1, y1)
            line.start = point1
            line.end = point2

            self.lines[i] = line
            self += point1
            self += point2

            if 0 <= x0 < self.width and 0 <= y0 < self.height:
                self.grid_matrix[y0, x0] = True
            if 0 <= x1 < self.width and 0 <= y1 < self.height:
                self.grid_matrix[y1, x1] = True


class Fractal:
    def __init__(
        self, atom: str, angle: float, start_rotation: float, rules: dict[str, str]
    ) -> None:
        self.atom = atom
        self.angle = angle
        self.rotation = start_rotation
        string_rules = rules
        self.rules: dict[str, list[str]] = {}

        self.atom = self.parse_rule(self.atom)

        for key, value in string_rules.items():
            self.rules[key] = self.parse_rule(value)

    def parse_rule(self, rule: str) -> list[str]:
        parsed_rule = list[str]()
        for char in rule:
            if char not in self.rules and re.match(r"[^A-Z\+\-\[\]\*\<\>]", char):
                raise ValueError("Invalid character in rule: " + char)
            parsed_rule.append(char)
        return parsed_rule

    def __call__(self, iterations: int):
        new_atom = self.atom
        for _ in range(iterations):
            temp_atom = list[str]()
            for symbol in new_atom:
                if symbol in self.rules:
                    temp_atom.extend(self.rules[symbol])
                else:
                    temp_atom.append(symbol)
            new_atom = temp_atom
        return new_atom

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(atom={self.atom}, angle={self.angle}, rotation={self.rotation}, rules={self.rules})"

    def rotate(self, angle: Number) -> None:
        self.rotation = (self.rotation + angle) % 360


if __name__ == "__main__":
    ITERATIONS = 9

    primary_color = "#111111"
    secondary_color = "#39dd57"


    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
    

    def gradient_generator(color1: str, color2: str, steps: int):
        r1, g1, b1 = hex_to_rgb(color1)
        r2, g2, b2 = hex_to_rgb(color2)

        for step in range(steps):
            t = step / (steps - 1)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            yield f"#{r:02x}{g:02x}{b:02x}"


    def apply_rules(frac: Fractal, frac_rules: Sequence[str], random_angle: float = 0.0) -> tuple[list[Point], list[Line]]:
        current_point = Point(0, 0)
        lines = list[Line]()
        points = [current_point]
        stack = list[tuple[Number, Point, int, int]]()
        gradient_colors = list(gradient_generator(primary_color, secondary_color, ITERATIONS + 1))
        current_color_index = 0
        line_width = 1

        for char in frac_rules:
            if re.match(r"[A-Z]", char):
                rad = np.radians(frac.rotation)
                new_x = current_point.x + line_length * np.cos(rad)
                new_y = current_point.y + line_length * np.sin(rad)
                new_point = Point(new_x, new_y)
                lines.append(Line(current_point, new_point))
                lines[-1]['color'] = gradient_colors[current_color_index]
                lines[-1]['width'] = line_width
                current_point = new_point
                points.append(current_point)

            elif char == "+":
                if random_angle:
                    frac.rotate(frac.angle + np.random.uniform(random_angle * -1, random_angle))
                else:
                    frac.rotate(frac.angle)

            elif char == "-":
                if random_angle:
                    frac.rotate(-frac.angle + np.random.uniform(random_angle * -1, random_angle))
                else:
                    frac.rotate(-frac.angle)

            elif char == "[":
                stack.append((frac.rotation, current_point, line_width, current_color_index))

            elif char == "]":
                frac.rotation, current_point, line_width, current_color_index = stack.pop()

            elif char == "*":
                current_color_index = min(ITERATIONS, current_color_index + 1)

            elif char == "<":
                line_width = max(1, line_width - 1)

            elif char == ">":
                line_width += 1

        return points, lines


    frac = Fractal(
        atom="A",
        angle=60,
        start_rotation=0,
        rules={
            "A": "B-A-B",
            "B": "A+B+A"
        }
    )
    print(frac)

    canvas = TkinterCanvas()
    print(canvas)

    frac_rules = list[str]()

    print("Fractal iterations: ")
    # --- Generate fractal ---
    for iteration in frac(ITERATIONS):
        print(iteration, end=" ")
        frac_rules.append(iteration)

    print("\nDrawing fractal...")
    line_length = 1

    points, lines = apply_rules(frac, frac_rules)

    center = Point(WIDTH // 2, HEIGHT // 2)
    frac_center = Point(0.0, 0.0)
    for point in points:
        frac_center.x += point.x
        frac_center.y += point.y

    frac_center.x //= len(points)
    frac_center.y //= len(points)

    xmax = max(point.x for point in points)
    ymax = max(point.y for point in points)
    xmin = min(point.x for point in points)
    ymin = min(point.y for point in points)
    scale_x = WIDTH / dx if (dx := xmax - xmin) != 0 else 1
    scale_y = HEIGHT / dy if (dy := ymax - ymin) != 0 else 1
    min_scale = min(scale_x, scale_y)

    matrix_scale = np.array(
        [
            [min_scale, 0, -xmin * scale_x],
            [0, min_scale, -ymin * scale_y],
            [0, 0, 1]
        ]
    )
    matrix_scale2 = np.array(
        [
            [0.8, 0, (WIDTH * (1 - 0.8)) // 2],
            [0, 0.8, (HEIGHT * (1 - 0.8)) // 2],
            [0, 0, 1],
        ]
    )

    matrix = matrix_scale2 @ matrix_scale
    print("Transformation matrix:\n", matrix)

    canvas.make_points(points)
    canvas.make_lines(lines)
    canvas.transform(matrix)

    for line in canvas.lines:
        canvas.tk_canvas.create_line(
            line.start.x, line.start.y, line.end.x, line.end.y, fill=line['color'], width=line['width']
        )
    # for point in canvas.points:
    #     canvas.tk_canvas.create_oval(
    #         point.x - 1, point.y - 1, point.x + 1, point.y + 1, fill="red"
    #     )

    print("Fractal drawn.")
    # print(canvas.lines)

    tkinter.mainloop()
