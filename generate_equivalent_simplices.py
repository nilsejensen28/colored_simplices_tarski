import logging
import numpy as np
import unittest

NUM_DIMENSIONS = 4
LOGGING_LEVEL = "INFO"
OUTPUT_FILE = "output.txt"


class Simplex_Tests(unittest.TestCase):

    def test_valid_simplex(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        s = Simplex([1, 2, 3], [1, 2, 2, 0])
        self.assertTrue(s.is_valid())

    def test_invalid_simplex(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        self.assertRaises(AssertionError, Simplex, [1, 2, 3], [1, 2, 2, 1])

    def test_valid_simplex_4_dimensions(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 4
        s = Simplex([1, 2, 3, 4], [1, 2, 3, 3, 0])
        self.assertTrue(s.is_valid())

    def test_invalid_simplex_4_dimensions(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 4
        self.assertRaises(AssertionError, Simplex, [
                          1, 2, 3, 4], [1, 2, 3, 3, 1])

    def test_equality_of_simplices(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        s1 = Simplex([1, 2, 3], [1, 2, 2, 0])
        s2 = Simplex([1, 2, 3], [1, 2, 2, 0])
        self.assertEqual(s1, s2)

    def test_hash_of_simplices(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        s1 = Simplex([1, 2, 3], [1, 2, 2, 0])
        s2 = Simplex([1, 2, 3], [1, 2, 2, 0])
        self.assertEqual(hash(s1), hash(s2))

    def test_double_color(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        s = Simplex([1, 2, 3], [1, 2, 2, 0])
        self.assertEqual(s.find_double_color(), 2)

    def test_generate_direct_neighboors_2_dimensions(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 2
        s = Simplex([1, 2], [1, 1, 0])
        neighboors = s.generate_direct_neighboors()
        neighboors_expected = {
            Simplex([2, 1], [1, 1, 0]), Simplex([2, 1], [1, 0, 0])}
        self.assertEqual(neighboors, neighboors_expected)

    def test_generate_direct_neighboors_3_dimensions(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        s = Simplex([1, 2, 3], [1, 2, 2, 0])
        neighboors = s.generate_direct_neighboors()
        neighboors_expected = {
            Simplex([1, 3, 2], [1, 2, 2, 0]), Simplex([1, 3, 2], [1, 2, 0, 0]), Simplex([2, 1, 3], [1, 1, 2, 0]), Simplex([2, 1, 3], [1, 2, 2, 0])}
        self.assertEqual(neighboors, neighboors_expected)

    def test_generate_equivalent_simplices_2_dimensions(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 2
        s = Simplex([1, 2], [1, 1, 0])
        visited = set()
        visited.add(s)
        equivalent_simplices = set()
        equivalent_simplices.add(s)
        recursive_generate_equivalent_simplices(
            s, visited, equivalent_simplices)
        equivalent_simplices_expected = {Simplex([1, 2], [1, 1, 0]), Simplex(
            [2, 1], [1, 0, 0]), Simplex([2, 1], [1, 1, 0])}
        self.assertEqual(equivalent_simplices, equivalent_simplices_expected)

    def test_generate_equivalent_simplices_3_dimensions(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        s = Simplex([1, 2, 3], [1, 2, 2, 0])
        visited = set()
        visited.add(s)
        equivalent_simplices = set()
        equivalent_simplices.add(s)
        recursive_generate_equivalent_simplices(
            s, visited, equivalent_simplices)
        self.assertEqual(len(equivalent_simplices), 16)

    def test_valid_face(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        f = Face([1, 2], [1, 2, 0])
        self.assertTrue(f.is_valid())

    def test_invalid_face(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        self.assertRaises(AssertionError, Face, [1, 2], [1, 2, 1])

    def test_generate_possible_smaller_simplices(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        f = Face([1, 2], [1, 2, 0])
        smaller_simplices = f.generate_possible_smaller_simplices()
        smaller_simplices_expected = {Simplex([3, 1, 2], [1, 1, 2, 0])}
        self.assertEqual(smaller_simplices, smaller_simplices_expected)

    def test_generate_possible_larger_simplices(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        f = Face([1, 2], [1, 2, 0])
        larger_simplices = f.generate_possible_larger_simplices()
        larger_simplices_expected = {Simplex([1, 2, 3], [1, 2, 0, 0])}
        self.assertEqual(larger_simplices, larger_simplices_expected)

    def test_equivalence_dictionary(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        equivalence_dictionary = generate_equivalence_dictionary()
        self.assertEqual(len(equivalence_dictionary), 65)

    def test_generate_neighboring_face_3_dimensions(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        f = Face([1, 2], [1, 2, 0])
        equivalence_dictionary = generate_equivalence_dictionary()
        neighbooring_smaller_faces, neighbooring_larger_faces = f.generate_neighbooring_faces(
            equivalence_dictionary)
        neighbooring_faces_expected = {
            Oriented_Face(Face([1, 2], [1, 2, 0]), False),
            Oriented_Face(Face([2, 1], [1, 2, 0]), True),
            Oriented_Face(Face([3, 2], [1, 2, 0]), True),
            Oriented_Face(Face([1, 3], [1, 2, 0]), True),
            Oriented_Face(Face([2, 3], [1, 2, 0]), False)}
        self.assertEqual(neighbooring_smaller_faces,
                         neighbooring_faces_expected)

    def test_generate_neighboring_face_2_dimensions(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 2
        f = Face([1], [1, 0])
        equivalence_dictionary = generate_equivalence_dictionary()
        neighbooring_smaller_faces, neighbooring_larger_faces = f.generate_neighbooring_faces(
            equivalence_dictionary)
        neighbooring_faces_expected = {
            Oriented_Face(Face([1], [1, 0]), True),
            Oriented_Face(Face([2], [1, 0]), False)}
        self.assertEqual(neighbooring_larger_faces,
                         neighbooring_faces_expected)

    def test_valid_path(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        path = Grid_Path([1, 2, -1, -2])
        self.assertTrue(path.is_valid())

    def test_invalid_path(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        self.assertRaises(AssertionError, Grid_Path, [1, 2, -1, -2, 4])

    def test_invalid_path_2(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        path = Grid_Path([1, 2, -1, -2, 1])
        self.assertFalse(path.is_valid())

    def test_is_cycle(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        path = Grid_Path([1, 2, -1, -2])
        self.assertTrue(path.is_cycle())

    def test_not_cycle(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        path = Grid_Path([1, 2, 1])
        self.assertFalse(path.is_cycle())

    def test_get_length(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        path = Grid_Path([1, 2, -1, -2])
        self.assertEqual(path.get_length(), 4)

    def test_find_a_simplex_path(self):
        global NUM_DIMENSIONS
        NUM_DIMENSIONS = 3
        path = Grid_Path([1, 2])
        oriented_face_start = Oriented_Face(Face([3, 2], [1, 2, 0]), True)
        oriented_face_end = Oriented_Face(Face([3, 2], [1, 2, 0]), True)
        result = []
        self.assertTrue(path.find_a_simplex_path(
            oriented_face_start, oriented_face_end, path=result))
        self.assertEqual(len(result), 2)


class Simplex():

    def __init__(self, dimensions, coloring):
        self.dimensions = dimensions
        self.coloring = coloring

        # Check the conditions for a simplex
        assert len(self.dimensions) == NUM_DIMENSIONS
        assert len(self.coloring) == NUM_DIMENSIONS + 1
        assert set(self.coloring) == set(range(0, NUM_DIMENSIONS))
        assert set(self.dimensions) == set(range(1, NUM_DIMENSIONS+1))

    def __eq__(self, other):
        # Check if two simplices are equivalent
        if self.coloring == other.coloring:
            if self.dimensions == other.dimensions:
                return True
        return False

    def find_double_color(self):
        # Find the double color
        for color in self.coloring:
            if self.coloring.count(color) == 2:
                return color
        else:
            raise ValueError("No double color found")

    def is_valid(self):
        # Check if the simplex is valid, that is if for every pair of indexes i, j such that i<j and coloring[i] > coloring[j] then j appears in indices i, ..., j-1
        for i in range(0, NUM_DIMENSIONS):
            for j in range(i+1, NUM_DIMENSIONS+1):
                if (self.coloring[i] > self.coloring[j] and self.coloring[j] != 0) or (self.coloring[i] == 0 and self.coloring[j] != 0):
                    found_i = False
                    for k in range(i, j):
                        if self.dimensions[k] == self.coloring[j]:
                            found_i = True
                            break
                    if not found_i:
                        logging.debug(
                            f"Simplex is not valid: {self.dimensions}, {self.coloring}, at index {i} we have color {self.coloring[i]} and at index {j} we have color {self.coloring[j]}, the dimension {self.coloring[j]} does not appear in the indices {i} to {j-1}")
                        return False
        return True

    def __hash__(self) -> int:
        return hash((tuple(self.dimensions), tuple(self.coloring)))

    def generate_direct_neighboors(self, same_cell=True):
        # Should return two neighbooing simplices
        double_color = self.find_double_color()
        neighboors = set()
        for i in range(1, NUM_DIMENSIONS):
            if self.coloring[i] == double_color:
                # Swap the dimensions i-1, i
                new_dimensions = self.dimensions.copy()
                new_dimensions[i-1], new_dimensions[i] = self.dimensions[i], self.dimensions[i-1]
                for color in range(0, NUM_DIMENSIONS):
                    # Check if the simplex is valid and add it to the list
                    new_coloring = self.coloring.copy()
                    new_coloring[i] = color
                    new_simplex = Simplex(new_dimensions, new_coloring)
                    if new_simplex.is_valid():
                        neighboors.add(new_simplex)
        if not same_cell:
            # Check if the first or last vertex are double colored
            if self.coloring[0] == double_color:
                new_dimensions = self.dimensions.copy()[1:]
                new_dimensions.append(self.dimensions[0])
                new_coloring = self.coloring.copy()[1:]
                new_coloring.append(self.coloring[0])
                # Test all colors for the last vertex
                for color in range(0, NUM_DIMENSIONS):
                    new_coloring[-1] = color
                    new_simplex = Simplex(new_dimensions, new_coloring.copy())
                    if new_simplex.is_valid():
                        neighboors.add(new_simplex)
            if self.coloring[NUM_DIMENSIONS] == double_color:
                new_dimensions = self.dimensions.copy()[:-1]
                new_dimensions.insert(0, self.dimensions[NUM_DIMENSIONS-1])
                new_coloring = self.coloring.copy()[:-1]
                new_coloring.insert(0, self.coloring[NUM_DIMENSIONS])
                # Test all colors for the first vertex
                for color in range(0, NUM_DIMENSIONS):
                    new_coloring[0] = color
                    new_simplex = Simplex(new_dimensions, new_coloring.copy())
                    if new_simplex.is_valid():
                        neighboors.add(new_simplex)
        return neighboors

    def __str__(self):
        line_1 = ""
        line_2 = ""
        for i in range(0, NUM_DIMENSIONS):
            line_1 += f"  {  self.dimensions[i]} "
            line_2 += f"{self.coloring[i]}-->"
        line_2 += f"{self.coloring[NUM_DIMENSIONS]}"
        return line_1 + "\n" + line_2 + "\n"

    def check_if_edge_simplex(self):
        # Check if removing first dimension yields a rainbow face
        new_coloring = self.coloring[1:]
        if len(set(new_coloring)) == NUM_DIMENSIONS:
            return True
        # Check if removing last dimension yields a rainbow face
        new_coloring = self.coloring[:-1]
        if len(set(new_coloring)) == NUM_DIMENSIONS:
            return True
        return False


class Face():
    missing_dimension: int
    coloring: list
    dimensions: list
    possible_smaller_simplices: set
    possible_larger_simplices: set

    def __init__(self, dimensions, coloring):
        # Dimensions is a list of the dimensions of the face
        # Coloring is a list of the colors of the vertices of the face
        self.dimensions = dimensions
        self.coloring = coloring

        # Sanity checks
        assert len(self.dimensions) == NUM_DIMENSIONS-1
        assert len(self.coloring) == NUM_DIMENSIONS
        assert set(self.coloring) == set(range(0, NUM_DIMENSIONS))

        missing_dimension = -1
        for i in range(1, NUM_DIMENSIONS+1):
            if i not in self.dimensions:
                missing_dimension = i
                break
        assert missing_dimension != -1
        self.missing_dimension = missing_dimension

        # Generate the possible smaller simplices
        self.possible_smaller_simplices = self.generate_possible_smaller_simplices()
        self.possible_larger_simplices = self.generate_possible_larger_simplices()

    def __eq__(self, other):
        if self.coloring == other.coloring:
            if self.dimensions == other.dimensions:
                return True
        return False

    def __hash__(self):
        return hash((tuple(self.dimensions), tuple(self.coloring)))

    def generate_possible_larger_simplices(self):
        # Generate all possible simplices that are smaller than this face
        # The simplices are generated by adding the missing dimension to the face
        simplices = set()
        for color in range(0, NUM_DIMENSIONS):
            new_coloring = self.coloring.copy()
            new_coloring.append(color)
            new_dimensions = self.dimensions.copy()
            new_dimensions.append(self.missing_dimension)
            simplex = Simplex(new_dimensions, new_coloring)
            if simplex.is_valid():
                simplices.add(simplex)
        return simplices

    def generate_possible_smaller_simplices(self):
        # Generate all possible simplices that are smaller than this face
        # The simplices are generated by removing one of the dimensions from the face
        simplices = set()
        for i in range(0, NUM_DIMENSIONS):
            new_coloring = self.coloring.copy()
            new_coloring.insert(0, i)
            new_dimensions = self.dimensions.copy()
            new_dimensions.insert(0, self.missing_dimension)
            simplex = Simplex(new_dimensions, new_coloring)
            if simplex.is_valid():
                simplices.add(simplex)
        return simplices

    def __str__(self):
        line_1 = ""
        line_2 = ""
        for i in range(0, NUM_DIMENSIONS-1):
            line_1 += f"  {  self.dimensions[i]} "
            line_2 += f"{self.coloring[i]}-->"
        line_2 += f"{self.coloring[NUM_DIMENSIONS-1]}"
        return line_1 + "\n" + line_2 + "\n"

    def generate_neighbooring_faces(self, equivalence_dictionary):
        neighbooring_smaller_faces = set()
        for simplex in self.possible_smaller_simplices:
            for equivalent_simplex in equivalence_dictionary[simplex]:
                new_coloring = equivalent_simplex.coloring.copy()[1:]
                if set(new_coloring) == set(range(0, NUM_DIMENSIONS)):
                    new_dimensions = equivalent_simplex.dimensions.copy()[1:]
                    new_face = Oriented_Face(
                        Face(new_dimensions, new_coloring), True)
                    if new_face.face.is_valid():
                        neighbooring_smaller_faces.add(new_face)
                new_coloring = equivalent_simplex.coloring.copy()[:-1]
                if set(new_coloring) == set(range(0, NUM_DIMENSIONS)):
                    new_dimensions = equivalent_simplex.dimensions.copy()[:-1]
                    new_face = Oriented_Face(
                        Face(new_dimensions, new_coloring), False)
                    if new_face.face.is_valid():
                        neighbooring_smaller_faces.add(new_face)
        neighbooring_larger_faces = set()
        for simplex in self.possible_larger_simplices:
            for equivalent_simplex in equivalence_dictionary[simplex]:
                new_coloring = equivalent_simplex.coloring.copy()[:-1]
                if set(new_coloring) == set(range(0, NUM_DIMENSIONS)):
                    new_dimensions = equivalent_simplex.dimensions.copy()[:-1]
                    new_face = Oriented_Face(
                        Face(new_dimensions, new_coloring), False)
                    if new_face.face.is_valid():
                        neighbooring_larger_faces.add(new_face)
                new_coloring = equivalent_simplex.coloring.copy()[1:]
                if set(new_coloring) == set(range(0, NUM_DIMENSIONS)):
                    new_dimensions = equivalent_simplex.dimensions.copy()[1:]
                    new_face = Oriented_Face(
                        Face(new_dimensions, new_coloring), True)
                    if new_face.face.is_valid():
                        neighbooring_larger_faces.add(new_face)
        return neighbooring_smaller_faces, neighbooring_larger_faces

    def is_valid(self):
        # Check if a face is valid
        for i in range(0, NUM_DIMENSIONS-1):
            for j in range(i+1, NUM_DIMENSIONS):
                if (self.coloring[i] > self.coloring[j] and self.coloring[j] != 0) or (self.coloring[i] == 0 and self.coloring[j] != 0):
                    found_i = False
                    for k in range(i, j):
                        if self.dimensions[k] == self.coloring[j]:
                            found_i = True
                            break
                    if not found_i:
                        logging.debug(
                            f"Face is not valid: {self.coloring}, {self.dimensions}, at index {i} we have color {self.coloring[i]} and at index {j} we have color {self.coloring[j]}, the dimension {j} does not appear in the indices {i} to {j-1}")
                        return False
        return True


class Oriented_Face():
    face: Face
    from_smaller_to_larger: bool

    def __init__(self, face, from_smaller_to_larger):
        self.face = face
        self.from_smaller_to_larger = from_smaller_to_larger

    def __eq__(self, other):
        if self.face == other.face and self.from_smaller_to_larger == other.from_smaller_to_larger:
            return True
        return False

    def __hash__(self):
        return hash((self.face, self.from_smaller_to_larger))

    def __str__(self):
        if self.from_smaller_to_larger:
            return f"{self.face} from smaller to larger"
        else:
            return f"{self.face} from larger to smaller"


class Cell():
    # A grid a large graph of simplices that are connected to each other
    # For every dimension we generate the set of simplices that are valid
    all_faces: set
    neighbooring_smaller_faces: dict
    neighbooring_larger_faces: dict

    def __init__(self):
        self.all_faces = generate_all_valid_faces()
        self.neighbooring_smaller_faces = {}
        self.neighbooring_larger_faces = {}
        self.equivalence_dictionary = generate_equivalence_dictionary()
        for face in self.all_faces:
            neighbooring_smaller_faces, neighbooring_larger_faces = face.generate_neighbooring_faces(
                self.equivalence_dictionary)
            self.neighbooring_smaller_faces[face] = neighbooring_smaller_faces
            self.neighbooring_larger_faces[face] = neighbooring_larger_faces

    def generate_equivalent_faces(self, face: Oriented_Face, side: int, side_2=None):
        # Generate all equivalent faces
        # Side represents on what side of the oriented face we want the equivalent faces
        # Side 2 is an optional parameter which we use to
        if side_2 is None:
            side_2 = 0
        allowed_smaller_simlices = face.face.possible_smaller_simplices
        allowed_larger_simplices = face.face.possible_larger_simplices
        if side > 0:
            # We want the larger simplices
            allowed_simplices = allowed_larger_simplices
        else:
            allowed_simplices = allowed_smaller_simlices
        equivalent_faces = set()
        for simplex in allowed_simplices:
            for equivalent_simplex in self.equivalence_dictionary[simplex]:
                # Try removing first dimension
                if side >= 0:
                    new_coloring = equivalent_simplex.coloring.copy()[1:]
                    new_dimensions = equivalent_simplex.dimensions.copy()[1:]
                    try:
                        new_face = Oriented_Face(
                            Face(new_dimensions, new_coloring), True)
                        if new_face.face.is_valid():
                            equivalent_faces.add(new_face)
                    except AssertionError:
                        pass
                # Try removing last dimension
                if side_2 <= 0:
                    new_coloring = equivalent_simplex.coloring.copy()[:-1]
                    new_dimensions = equivalent_simplex.dimensions.copy()[:-1]
                    try:
                        new_face = Oriented_Face(
                            Face(new_dimensions, new_coloring), False)
                        if new_face.face.is_valid():
                            equivalent_faces.add(new_face)
                    except AssertionError:
                        pass
        return equivalent_faces

    def __copy__(self):
        new_cell = Cell()
        new_cell.all_faces = self.all_faces.copy()
        new_cell.neighbooring_smaller_faces = self.neighbooring_smaller_faces.copy()
        new_cell.neighbooring_larger_faces = self.neighbooring_larger_faces.copy()
        new_cell.equivalence_dictionary = self.equivalence_dictionary.copy()
        return new_cell

    def allowed_paths(self, dimension1: int, dimensions2):
        logging.debug(
            f"Computing allowed paths from dimension {dimension1} to {dimensions2}")
        faces_on_side_1 = set()
        faces_on_side_2 = set()
        pairs_of_faces = set()
        for face in self.all_faces:
            if int(abs(dimension1)) == face.missing_dimension:
                faces_on_side_1.add(face)
            if int(abs(dimensions2)) == face.missing_dimension:
                faces_on_side_2.add(face)
        if dimension1 > 0:
            dimension_1_s_l = True
        else:
            dimension_1_s_l = False
        if dimensions2 > 0:
            dimension_2_s_l = True
        else:
            dimension_2_s_l = False
        for face_1 in faces_on_side_1:
            logging.debug(f"Face 1: {face_1}")
            for face_2 in faces_on_side_2:
                logging.debug(f"Face 2: {face_2}")
                oriented_face_1 = Oriented_Face(face_1, dimension_1_s_l)
                oriented_face_2 = Oriented_Face(face_2, dimension_2_s_l)
                equivalent_faces_1 = self.generate_equivalent_faces(
                    oriented_face_1, dimension1, int(dimensions2/abs(dimensions2)))
                if oriented_face_2 in equivalent_faces_1:
                    pairs_of_faces.add((oriented_face_1, oriented_face_2))
                    logging.debug(
                        f"Found pair of faces: {oriented_face_1}, {oriented_face_2}")
        return pairs_of_faces


class Grid():

    def __init__(self, side_length: int, cell: Cell):
        self.side_length = side_length
        self.cell = cell
        # A n dimensional array of cells
        shape = tuple([side_length for i in range(0, NUM_DIMENSIONS)])
        array = np.ndarray(shape=shape, dtype=Cell)
        array = array.flatten()
        for i in range(0, len(array)):
            array[i] = cell.__copy__()
        array = array.reshape(shape)
        self.array = array
        print(self.array)


class Grid_Path():
    dimensions: list
    cell: Cell

    def __init__(self, dimensions, cell=None):
        self.dimensions = dimensions
        # Sanity checks
        allowed_dimensions = set(
            range(1, NUM_DIMENSIONS+1)).union(set(range(-NUM_DIMENSIONS, 0)))
        for dimension in self.dimensions:
            assert dimension in allowed_dimensions
        if cell is None:
            self.cell = Cell()
        else:
            self.cell = cell

    def is_valid(self):
        # Should not self intersect
        current_position = [0 for i in range(0, NUM_DIMENSIONS)]
        visited_positions = set()
        visited_positions.add(tuple(current_position))
        for i in range(0, len(self.dimensions)):
            dimension = int(abs(self.dimensions[i]))
            direction = int(self.dimensions[i]/dimension)
            current_position[dimension -
                             1] = current_position[dimension-1] + direction
            if tuple(current_position) in visited_positions:
                if i == len(self.dimensions)-1:
                    if tuple(current_position) != tuple([0 for i in range(0, NUM_DIMENSIONS)]):
                        return False
                else:
                    return False
            visited_positions.add(tuple(current_position))
        return True

    def is_cycle(self):
        # Sanity checks
        # Each time that dimension i appears, dimension -i should also appear (the same number of times)
        is_cycle = self.is_valid()
        for i in range(1, NUM_DIMENSIONS+1):
            count_i = self.dimensions.count(i)
            count_minus_i = self.dimensions.count(-i)
            if not count_i == count_minus_i:
                is_cycle = False
                break
        return is_cycle

    def get_length(self):
        return len(self.dimensions)

    def find_a_simplex_path(self, starting_face: Oriented_Face, ending_face: Oriented_Face, path):
        # Sanity check that the path is valid
        if not self.is_valid():
            raise ValueError("Path is not valid")

        if self.get_length() == 0:
            direction = -1
            if starting_face.from_smaller_to_larger:
                direction = 1
            if ending_face in self.cell.generate_equivalent_faces(starting_face, direction):
                path = [starting_face, ending_face]
                return True, path
            else:
                return False, []
        else:
            new_path = Grid_Path(self.dimensions[:-1])
            # Compute dimension of the last face
            last_dimension = ending_face.face.missing_dimension
            if not ending_face.from_smaller_to_larger:
                last_dimension = -last_dimension
            allowed_pairs = self.cell.allowed_paths(
                self.dimensions[-1], last_dimension)
            logging.debug(
                f"Looking for pairs from dimension {self.dimensions[-1]} to {last_dimension}")
            found_path = False
            for pair in allowed_pairs:
                if pair[1] == ending_face:
                    found_path, path_previous = new_path.find_a_simplex_path(
                        starting_face, pair[0], path)
                    if found_path:
                        path_previous.append(pair[0])
                        break
            if found_path:
                path = path_previous
                return True, path
            else:
                return False, []

    def __str__(self):
        return str(self.dimensions)


def valid_combination_of_simplices(simplex1, simplex2):
    # A combination of simplices is invalid, if they have different colors for the same vertex
    # Check the first vertex
    if simplex1.coloring[0] != simplex2.coloring[0]:
        return False
    for i in range(0, NUM_DIMENSIONS):
        if simplex1.dimensions[i] == simplex2.dimensions[i]:
            if simplex1.coloring[i+1] != simplex2.coloring[i+1]:
                return False
        else:
            break
    if simplex1.coloring[NUM_DIMENSIONS] != simplex2.coloring[NUM_DIMENSIONS]:
        return False
    for i in range(NUM_DIMENSIONS-1, 0, -1):
        if simplex1.dimensions[i] == simplex2.dimensions[i]:
            if simplex1.coloring[i] != simplex2.coloring[i]:
                return False
        else:
            break
    return True


def recursive_generate_equivalent_simplices(simplex, visited, equivalent_simplices):
    # Generate all equivalent simplices
    # Visited represents all simplices that are on the path from the root simplex to the current simplex
    # equivalent_simplices represents all equivalent simplices it a set
    # First check if we are valid ourselves
    if not simplex.is_valid():
        return
    neighboors = simplex.generate_direct_neighboors()
    for neighboor in neighboors:
        if neighboor not in visited:
            visited.add(neighboor)
            # Check if neighboor is valid with respect to the other simplices on the path from s1 to neighboor
            valid = True
            for visited_simplex in visited:
                if not valid_combination_of_simplices(neighboor, visited_simplex):
                    valid = False
                    break
            if valid:
                equivalent_simplices.add(neighboor)
                recursive_generate_equivalent_simplices(
                    neighboor, visited, equivalent_simplices)
            visited.remove(neighboor)


def generate_all_valid_simplices():
    count_all_simplices = 0
    count_valid_simplices = 0
    simplices = set()
    permutations_dimensions = generate_permutations(NUM_DIMENSIONS)
    for permutation_dim in permutations_dimensions:
        # Generate all possible colorings
        permutations_colorings = generate_permutations(NUM_DIMENSIONS+1)
        for permutation in permutations_colorings:
            # Replace NUM_DIMENSIONS+1 with 0
            permutation[permutation.index(NUM_DIMENSIONS+1)] = 0
            for color in range(0, NUM_DIMENSIONS):
                coloring = permutation.copy()
                # replace the color NUM_DIMENSIONS with color
                coloring[coloring.index(NUM_DIMENSIONS)] = color
                simplex = Simplex(
                    permutation_dim, coloring)
                count_all_simplices += 1
                if simplex.is_valid():
                    count_valid_simplices += 1
                    simplices.add(simplex)
    logging.debug(
        f"Generated {count_all_simplices} simplices, of which {count_valid_simplices} are valid")
    return simplices


def generate_all_valid_faces():
    count_all_faces = 0
    count_valid_faces = 0
    faces = set()
    for missing_dimension in range(1, NUM_DIMENSIONS+1):
        permutations_dimensions = generate_permutations(NUM_DIMENSIONS)
        for permutation_dim in permutations_dimensions:
            # Generate all possible colorings
            # Remove the missing dimension from the dimensions
            permutation_dim.remove(missing_dimension)
            permutations_colorings = generate_permutations(NUM_DIMENSIONS)
            for permutation in permutations_colorings:
                # Replace NUM_DIMENSIONS with 0
                coloring = permutation.copy()
                coloring[coloring.index(NUM_DIMENSIONS)] = 0
                face = Face(
                    permutation_dim, coloring)
                count_all_faces += 1
                if face.is_valid():
                    count_valid_faces += 1
                    faces.add(face)
                else:
                    logging.debug(f"Face is not valid: {face}")
    logging.info(
        f"Generated {count_all_faces} faces, of which {count_valid_faces} are valid")
    return faces


def check_existence_of_path(simplex1, simplex2):
    visited = set()
    visited.add(simplex1)
    equivalent_simplices = set()
    equivalent_simplices.add(simplex1)
    recursive_generate_equivalent_simplices(
        simplex1, visited, equivalent_simplices)
    return simplex2 in equivalent_simplices


def generate_equivalence_dictionary():
    all_simplices = generate_all_valid_simplices()
    equivalence_dictionary = {}
    for simplex in all_simplices:
        equivalence_dictionary[simplex] = set()
        visited = set()
        visited.add(simplex)
        equivalent_simplices = set()
        recursive_generate_equivalent_simplices(
            simplex, visited, equivalent_simplices)
        for equivalent_simplex in equivalent_simplices:
            equivalence_dictionary[simplex].add(equivalent_simplex)
    return equivalence_dictionary


def generate_permutations(n: int):
    if n == 1:
        return [[1]]
    else:
        permutations = []
        for permutation in generate_permutations(n-1):
            for i in range(n-1, -1, -1):
                new_permutation = permutation.copy()
                new_permutation.insert(i, n)
                permutations.append(new_permutation)
        return permutations


def generate_permutations_from_list(list: list):
    permutations = generate_permutations(len(list))
    for permutation in permutations:
        for i in range(0, len(permutation)):
            permutation[i] = list[permutation[i]-1]
    return permutations


def generate_cycle_from_missing_face(face: Oriented_Face, cell=None):
    logging.debug(f"Generating cycle from the oriented face: {face}")
    missing_dimension = face.face.missing_dimension
    if face.from_smaller_to_larger:
        direction = 1
    else:
        direction = -1
    list_of_dimensions = [
        i for i in range(-NUM_DIMENSIONS, NUM_DIMENSIONS+1, 1)]
    list_of_dimensions.remove(0)
    list_of_dimensions.remove(direction*missing_dimension)
    if len(list_of_dimensions) != 2*NUM_DIMENSIONS-1:  # Sanity check
        logging.error(
            f"List of dimensions is not of the correct length: {list_of_dimensions}")
        raise ValueError("List of dimensions is not of the correct length")
    permutations = generate_permutations_from_list(list_of_dimensions)
    set_of_cycles = set()
    for permutation in permutations:
        cycle = Grid_Path(permutation, cell=cell)
        if cycle.is_valid():
            set_of_cycles.add(cycle)
        else:
            logging.debug(f"Cycle is not valid: {cycle}")
    return set_of_cycles


def test_all_cycles():
    cell = Cell()
    reset_output_file(OUTPUT_FILE)
    for face in cell.all_faces:
        for orientation in [True, False]:
            oriented_face = Oriented_Face(face, orientation)
            cycles = generate_cycle_from_missing_face(oriented_face, cell=cell)
            for cycle in cycles:
                logging.debug(f"Testing cycle: {cycle}")
                path_exists, result = cycle.find_a_simplex_path(
                    oriented_face, oriented_face, [])
                if not path_exists:
                    logging.debug(f"Cycle: {cycle} can not be realized")
                    print_line_to_output_file(
                        OUTPUT_FILE, f"Cycle: {cycle} cannot be realized with the oriented face:\n {oriented_face}\n")
                else:
                    logging.info(f"Cycle: {cycle} can be realized")
                    print_line_to_output_file(
                        OUTPUT_FILE, f"Cycle: {cycle} can be realized with the oriented face:\n {oriented_face}\n")
                    for face in result:
                        logging.info(face)
                        print_line_to_output_file(OUTPUT_FILE, str(face))
                    logging.info("End of cycle")


def reset_output_file(output_file):
    with open(output_file, "w") as file:
        file.write("")
    logging.info(f"Output file {output_file} has been reset")


def print_line_to_output_file(output_file, line):
    with open(output_file, "a") as file:
        file.write(line + "\n")


def main():
    # Set the logging level
    logging.basicConfig(level=LOGGING_LEVEL)
    test_all_cycles()


if __name__ == "__main__":
    main()
