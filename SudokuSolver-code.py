import numpy as np

# Load sudokus
sudoku = np.load("data/very_easy_puzzle.npy")
print("very_easy_puzzle.npy has been loaded into the variable sudoku")
print(f"sudoku.shape: {sudoku.shape}, sudoku[0].shape: {sudoku[0].shape}, sudoku.dtype: {sudoku.dtype}")


# importing all the required modules 
import copy # used to create a copy of the current game state
import numpy as np # used to handle the inputted and outputted sudoku game states 
from itertools import combinations # used for iterating through possible combinations

# a class used to represent each sudoku state
class PartialSudokuState:

    # a function that instantiates the intial board setup
    def __init__(self, initial_config):
        self.possible_values = [[[i for i in range(1, 10)] for _ in range(0, 9)] for _ in range(0, 9)]
        self.final_values = [[-1] * 9 for _ in range(0, 9)]

        for row in range(0, 9):
                for column in range(0, 9):
                    if initial_config[row][column] != 0:
                        state = self.set_value(row, column, initial_config[row][column])  
                        self.possible_values = state.possible_values
                        self.final_values = state.final_values

 
        self.naked_candidates()
        self.intersection_removal()
        if not self.AC3():      
            self.possible_values = [[[] for _ in range(0, 9)] for _ in range(0, 9)]
            self.final_values = [[-1] * 9 for _ in range(0,9)]    
   
    # a function for determining whether or not the current state is complete
    def is_solved(self):
        return all(value != -1 for row in self.final_values for value in row)

    # a function for determining whether or not the current state is invalid
    def is_invalid(self):
        return any(len(values) == 0 for row in self.possible_values for values in row)

    # a function that copies the current game state and updates it in line with the new move 
    def set_value(self, row, column, value):
        state = copy.deepcopy(self)
            
        if value not in self.possible_values[row][column]:
            state.possible_values = [[[] for _ in range(0, 9)] for _ in range(0, 9)]
            state.final_values = [[-1] * 9 for _ in range(0,9)]
            return state
                
        state.possible_values[row][column] = [value]
        state.final_values[row][column] = value

        for update_col in range(0, column):
            if value in state.get_possible_values(row, update_col):
                state.possible_values[row][update_col].remove(value)

        for update_col in range(column+1, 9):
            if value in state.get_possible_values(row, update_col):
                state.possible_values[row][update_col].remove(value)

        for update_row in range(0, row):
            if value in state.get_possible_values(update_row, column):
                state.possible_values[update_row][column].remove(value)

        for update_row in range(row+1, 9):
            if value in state.get_possible_values(update_row, column):
                state.possible_values[update_row][column].remove(value)

        square_column = (column // 3)
        min_col = int(square_column * 3)
        max_col = int((square_column + 1) * 3)

        square_row = (row // 3)
        min_row = int(square_row * 3)
        max_row = int((square_row + 1) * 3)

        for update_col in range(min_col, max_col):
            for update_row in range(min_row, max_row):
                if (update_col != column) and (update_row != row) and (value in state.get_possible_values(update_row, update_col)):
                    state.possible_values[update_row][update_col].remove(value)
    
        
        singleton_columns = state.get_singleton_columns()
        while len(singleton_columns) > 0:
            row, col = singleton_columns[0]
            state    = state.set_value(row, col, state.possible_values[row][col][0])
            singleton_columns = state.get_singleton_columns()

        return state

    # a constraint satisfaction that helps prune the number of possible game states
    def naked_candidates(self):

        #rows
        for row in range(0, 9):
            remaining_values = []
            cols = []
            for column in range(0, 9):
                if len(self.get_possible_values(row, column)) > 1:
                    remaining_values.append(set(self.get_possible_values(row, column)))
                    cols.append(column)
            
            # doubles
            doubles = combinations(range(len(remaining_values)), 2)
            for idx1, idx2 in doubles:
                double1 = remaining_values[idx1]
                double2 = remaining_values[idx2]
                union_values = double1.union(double2)
                #check for naked double 
                if len(union_values) == 2:
                    for update_col in range(0, 9):
                        if (update_col != cols[idx1]) and (update_col != cols[idx2]):
                            values = set(self.get_possible_values(row, update_col))
                            for val_to_remove in union_values:
                                if val_to_remove in values:
                                    self.possible_values[row][update_col].remove(val_to_remove)
                                
            # triples
            triples = combinations(range(len(remaining_values)), 3)
            for index1, index2, index3 in triples:
                triple1 = remaining_values[index1]
                triple2 = remaining_values[index2]
                triple3 = remaining_values[index3]
                union_values = triple1.union(triple2, triple3)
                #check for naked triple
                if len(union_values) == 3:
                    for update_col in range(0, 9):
                        if (update_col != cols[index1]) and (update_col != cols[index2]) and (update_col != cols[index3]):
                            values = self.get_possible_values(row, update_col)
                            for val_to_remove in union_values:
                                if val_to_remove in values:
                                    self.possible_values[row][update_col].remove(val_to_remove)
                                
            #quads
            quads = combinations(range(len(remaining_values)), 4)
            for index1, index2, index3, index4 in quads:
                quad1 = remaining_values[index1]
                quad2 = remaining_values[index2]
                quad3 = remaining_values[index3]
                quad4 = remaining_values[index4]
                union_values = quad1.union(quad2, quad3, quad4)
                #check for naked quad
                if len(union_values) == 4:
                    for update_col in range(0, 9):
                        if (update_col != cols[index1]) and (update_col != cols[index2]) and (update_col != cols[index3]) and (update_col != cols[index4]):
                            values = self.get_possible_values(row, update_col)
                            for val_to_remove in union_values:
                                if val_to_remove in values:
                                    self.possible_values[row][update_col].remove(val_to_remove)
                                    
                          
        # columns
        for column in range(0, 9):
            remaining_values = []
            rows = []
            for row in range(0, 9):
                if len(self.get_possible_values(row, column)) > 1:
                    remaining_values.append(set(self.get_possible_values(row, column)))
                    rows.append(row)
            
            # doubles
            doubles = combinations(range(len(remaining_values)), 2)
            for index1, index2 in doubles:
                double1 = remaining_values[index1]
                double2 = remaining_values[index2]
                union_values = double1.union(double2)
                #check for naked double 
                if len(union_values) == 2:
                    for update_row in range(0, 9):
                        if (update_row != rows[index1]) and (update_row != rows[index2]):
                            values = self.get_possible_values(update_row, column)
                            for val_to_remove in union_values:
                                if val_to_remove in values:
                                    self.possible_values[update_row][column].remove(val_to_remove)
                                         
            # triples
            triples = combinations(range(len(remaining_values)), 3)
            for index1, index2, index3 in triples:
                triple1 = remaining_values[index1]
                triple2 = remaining_values[index2]
                triple3 = remaining_values[index3]
                union_values = triple1.union(triple2, triple3)
                #check for naked triple
                if len(union_values) == 3:
                    for update_row in range(0, 9):
                        if (update_row != rows[index1]) and (update_row != rows[index2]) and (update_row != rows[index3]):
                            values = self.get_possible_values(update_row, column)
                            for val_to_remove in union_values:
                                if val_to_remove in values:
                                    self.possible_values[update_row][column].remove(val_to_remove)
                               
            #quads
            quads = combinations(range(len(remaining_values)), 4)
            for index1, index2, index3, index4 in quads:
                quad1 = remaining_values[index1]
                quad2 = remaining_values[index2]
                quad3 = remaining_values[index3]
                quad4 = remaining_values[index4]
                union_values = quad1.union(quad2, quad3, quad4)
                #check for naked quad
                if len(union_values) == 4:
                    for update_row in range(0, 9):
                        if (update_row != rows[index1]) and (update_row != rows[index2]) and (update_row != rows[index3]) and (update_row != rows[index4]):
                            values = self.get_possible_values(update_row, column)
                            for val_to_remove in union_values:
                                if val_to_remove in values:
                                    self.possible_values[update_row][column].remove(val_to_remove)
                            

        #squares
        for square_row in range(0, 3):
            for square_col in range(0, 3):
                remaining_values = []
                cell_indices = []
                for row in range((square_row * 3), ((square_row + 1) * 3)):
                    for column in range((square_col * 3), ((square_col + 1) * 3)):
                        if len(self.get_possible_values(row, column)) > 1:
                            remaining_values.append(set(self.get_possible_values(row, column)))
                            cell_indices.append((row, column))
                      
                # doubles
                doubles = combinations(range(len(remaining_values)), 2)
                for index1, index2 in doubles:
                    double1 = remaining_values[index1]
                    double2 = remaining_values[index2]
                    union_values = double1.union(double2)
                    #check for naked double 
                    if len(union_values) == 2:
                        for index, (update_row, update_col) in enumerate(cell_indices):
                                if (index != index1) and (index != index2):
                                    values = self.get_possible_values(update_row, update_col)
                                    for val_to_remove in union_values:
                                        if val_to_remove in values:
                                            self.possible_values[update_row][update_col].remove(val_to_remove)
                                                       
                # triples
                triples = combinations(range(len(remaining_values)), 3)
                for index1, index2, index3 in triples:
                    triple1 = remaining_values[index1]
                    triple2 = remaining_values[index2]
                    triple3 = remaining_values[index3]
                    union_values = triple1.union(triple2, triple3)
                    #check for naked triple 
                    if len(union_values) == 3:
                        for index, (update_row, update_col) in enumerate(cell_indices):
                                if (index != index1) and (index != index2) and (index != index3):
                                    values = self.get_possible_values(update_row, update_col)
                                    for val_to_remove in union_values:
                                        if val_to_remove in values:
                                            self.possible_values[update_row][update_col].remove(val_to_remove)
                                        
                # quads
                quads = combinations(range(len(remaining_values)), 4)
                for index1, index2, index3, index4 in quads:
                    quad1 = remaining_values[index1]
                    quad2 = remaining_values[index2]
                    quad3 = remaining_values[index3]
                    quad4 = remaining_values[index4]
                    union_values = quad1.union(quad2, quad3, quad4)
                    #check for naked quad 
                    if len(union_values) == 4:
                        for index, (update_row, update_col) in enumerate(cell_indices):
                                if (index != index1) and (index != index2) and (index != index3) and (index != index4):
                                    values = self.get_possible_values(update_row, update_col)
                                    for val_to_remove in union_values:
                                        if val_to_remove in values:
                                           self.possible_values[update_row][update_col].remove(val_to_remove)
                                         
    # a constraint satisfaction that helps prune the number of possible game states
    def hidden_candidates(self):

        # rows
        for row in range(0, 9):
            no_of_values = {}
            for column in range(0,9):
                    for value in self.get_possible_values(row, column):
                        if value in no_of_values:
                            no_of_values[value].add(column)
                        else:
                            no_of_values[value] = set([column])

            # doubles
            d_no_of_values = {value: positions for value, positions in no_of_values.items() if len(positions) > 1}
            doubles = combinations(d_no_of_values.items(), 2)
            for (double1, double2) in doubles:
                value1, positions1 = double1
                value2, positions2 = double2
                union_values = positions1.union(positions2)
                # hidden double found within a given row
                if len(union_values) == 2:
                    for index in union_values:
                        for value in self.get_possible_values(row, index):
                            if (value != value1) and (value != value2):
                                self.possible_values[row][index].remove(value)
                        
            # triples
            t_no_of_values = {value: positions for value, positions in no_of_values.items() if len(positions) > 2}
            triples = combinations(t_no_of_values.items(), 3)
            for (triple1, triple2, triple3) in triples:
                value1, positions1 = triple1
                value2, positions2 = triple2
                value3, positions3 = triple3
                union_values = positions1.union(positions2, positions3)
                # hidden triple found within a given row
                if len(union_values) == 3:
                    for index in union_values:
                        for value in self.get_possible_values(row, index):
                            if (value != value1) and (value != value2) and (value != value3):
                                self.possible_values[row][index].remove(value)

            # quads
            q_no_of_values = {value: positions for value, positions in no_of_values.items() if len(positions) > 3}
            quads = combinations(q_no_of_values.items(), 4)
            for (quad1, quad2, quad3, quad4) in quads:
                value1, positions1 = quad1
                value2, positions2 = quad2
                value3, positions3 = quad3
                value4, positions4 = quad4
                union_values = positions1.union(positions2, positions3, positions4)
                # hidden quad found within a given row
                if len(union_values) == 4:
                    for index in union_values:
                       for value in self.get_possible_values(row, index):
                           if (value != value1) and (value != value2) and (value != value3) and (value != value4):
                               self.possible_values[row][index].remove(value)


        # columns
        for column in range(0, 9):
            no_of_values = {}
            for row in range(0,9):
                for value in self.get_possible_values(row, column):
                    if value in no_of_values:
                        no_of_values[value].add(row)
                    else:
                        no_of_values[value] = set([row])

            # doubles
            d_no_of_values = {value: positions for value, positions in no_of_values.items() if len(positions) > 1}
            doubles = combinations(d_no_of_values.items(), 2)
            for (double1, double2) in doubles:
                value1, positions1 = double1
                value2, positions2 = double2
                union_values = positions1.union(positions2)
                # hidden double found within a given column
                if len(union_values) == 2:
                    for index in union_values:
                        for value in self.get_possible_values(index, column):
                            if (value != value1) and (value != value2):
                                self.possible_values[index][column].remove(value)
                        
                    
            # triples
            t_no_of_values = {value: positions for value, positions in no_of_values.items() if len(positions) > 2}
            triples = combinations(t_no_of_values.items(), 3)
            for (triple1, triple2, triple3) in triples:
                value1, positions1 = triple1
                value2, positions2 = triple2
                value3, positions3 = triple3
                union_values = positions1.union(positions2, positions3)
                # hidden triple found within a given column
                if len(union_values) == 3:
                    for index in union_values:
                        for value in self.get_possible_values(index, column):
                            if (value != value1) and (value != value2) and (value != value3):
                                self.possible_values[index][column].remove(value)
                        

            # quads
            q_no_of_values = {value: positions for value, positions in no_of_values.items() if len(positions) > 3}
            quads = combinations(q_no_of_values.items(), 4)
            for (quad1, quad2, quad3, quad4) in triples:
                value1, positions1 = quad1
                value2, positions2 = quad2
                value3, positions3 = quad3
                value4, positions4 = quad4
                union_values = positions1.union(positions2, positions3, positions4)
                # hidden quad found within a given column
                if len(union_values) == 4:
                    for index in union_values:
                        for value in self.possible_values(index, column):
                            if (value != value1) and (value != value2) and (value != value3) and (value != value4):
                                self.possible_values[index][column].remove(value)

                        

        # squares
        for square_row in range(0, 3):
            for square_col in range(0, 3):
                no_of_values = {}
                for row in range((square_row * 3), ((square_row + 1) * 3)):
                    for column in range((square_col * 3), ((square_col + 1) * 3)):
                        for value in self.get_possible_values(row, column):
                            if value in no_of_values:
                                no_of_values[value].add((row, column))
                            else:
                                no_of_values[value] = set([(row, column)])
            

                # doubles
                d_no_of_values = {value: positions for value, positions in no_of_values.items() if len(positions) > 1}
                doubles = combinations(d_no_of_values.items(), 2)
                for (double1, double2) in doubles:
                    value1, positions1 = double1
                    value2, positions2 = double2
                    union_values = positions1.union(positions2)
                    # hidden double found in a given square
                    if len(union_values) == 2:
                        for (row, col) in union_values:
                            for value in self.get_possible_values(row, col):
                                if (value != value1) and (value != value2):
                                    self.possible_values[row][col].remove(value)

                # triples 
                t_no_of_values = {value: positions for value, positions in no_of_values.items() if len(positions) > 2}
                triples = combinations(t_no_of_values.items(), 3)
                for (triple1, triple2, triple3) in triples:
                    value1, positions1 = triple1
                    value2, positions2 = triple2
                    value3, positions3 = triple3
                    union_values = positions1.union(positions2, positions3)
                    # hidden double found in a given square
                    if len(union_values) == 3:
                        for (row, col) in union_values:
                            for value in self.get_possible_values(row, col):
                                if (value != value1) and (value != value2) and (value != value3):
                                    self.possible_values[row][col].remove(value)

                # quads
                q_no_of_values = {value: positions for value, positions in no_of_values.items() if len(positions) > 3}
                quads = combinations(q_no_of_values.items(), 4)
                for (quad1, quad2, quad3, quad4) in quads:
                    value1, positions1 = quad1
                    value2, positions2 = quad2
                    value3, positions3 = quad3
                    value4, positions4 = quad4
                    union_values = positions1.union(positions2, positions3, positions4)
                    # hidden quad found in a given square
                    if len(union_values) == 4:
                        for (row, col) in union_values:
                            for value in self.get_possible_values(row, col):
                                if (value != value1) and (value != value2) and (value != value3) and (value != value4):
                                    self.possible_values[row][col].remove(value)
                        
    # a constraint satisifaction that helps prune the number of possible game states
    def intersection_removal(self):
        
        # rows
        for row in range(0, 9):
            no_of_values = {}
            for column in range(0,9):
                for value in self.get_possible_values(row, column):
                    if value in no_of_values:
                        no_of_values[value].add(column)
                    else:
                        no_of_values[value] = set([column])
         
            d_no_of_values = {value: positions for value, positions in no_of_values.items() if (len(positions) == 2) or (len(positions) == 3)}
            for (num, positions) in (d_no_of_values.items()):
                positions = list(positions)
                if (len(positions) == 2):

                    if (positions[0] // 3) == (positions[1] // 3):
                        # two possible locations for a given value in a row appear in the same box
                        box_row = row // 3
                        box_col = (positions[0] // 3)
                        for update_row in range((box_row * 3), ((box_row + 1) * 3)):
                            for update_col in range((box_col * 3), ((box_col + 1) * 3)):
                                if (update_row != row) and (update_col not in positions):
                                    if num in self.get_possible_values(update_row, update_col):
                                        self.possible_values[update_row][update_col].remove(num)


                if (len(positions) == 3):
                    if (positions[0] // 3) == (positions[1] // 3) == (positions[2] // 3):
                        #three possible positions for a given value in a row appear in the same box
                        box_row = row // 3
                        box_col = (positions[0] // 3)
                        for update_row in range((box_row * 3), ((box_row + 1) * 3)):
                            for update_col in range((box_col * 3), ((box_col + 1) * 3)):
                                if (update_row != row) and (update_col not in positions):
                                    if num in self.get_possible_values(update_row, update_col):
                                        self.possible_values[update_row][update_col].remove(num)
                    
        # columns
        for column in range(0, 9):
            no_of_values = {}
            for row in range(0,9):
                for value in self.get_possible_values(row, column):
                    if value in no_of_values:
                        no_of_values[value].add(row)
                    else:
                        no_of_values[value] = set([row])
         
            d_no_of_values = {value: positions for value, positions in no_of_values.items() if (len(positions) == 2) or (len(positions) == 3)}
            for num, positions in d_no_of_values.items():
                positions = list(positions)
                if (len(positions) == 2):

                    if (positions[0] // 3) == (positions[1] // 3):
                        # two possible positions for a given value in a a column appear in the same box
                        box_row = (positions[0] // 3)
                        box_col = column // 3
                        for update_row in range((box_row * 3), ((box_row + 1) * 3)):
                            for update_col in range((box_col * 3), ((box_col + 1) * 3)):
                                if (update_row not in positions) and (update_col != column):
                                    if num in self.get_possible_values(update_row, update_col):
                                        self.possible_values[update_row][update_col].remove(num)


                if (len(positions) == 3):
                    if (positions[0] // 3) == (positions[1] // 3) == (positions[2] // 3):
                        # three possible positions for a given value in a column appear in the same box
                        box_row = (positions[0] // 3)
                        box_col = column // 3
                        for update_row in range((box_row * 3), ((box_row + 1) * 3)):
                            for update_col in range((box_col * 3), ((box_col + 1) * 3)):
                                if (update_row not in positions) and (update_col != column):
                                    if num in self.get_possible_values(update_row, update_col):
                                        self.possible_values[update_row][update_col].remove(num)

        # boxes
        for box_row in range(0, 3):
            for box_col in range(0, 3):
                no_of_values = {}
                for row in range((box_row * 3), ((box_row+1) * 3)):
                    for col in range((box_col * 3), ((box_col + 1) * 3)):
                        for value in self.get_possible_values(row, col):
                            if value in no_of_values:
                                no_of_values[value].add((row, col))
                            else:
                                no_of_values[value] = set([(row,col)])
                
                d_no_of_values = {value: positions for value, positions in no_of_values.items() if (len(positions) == 2) or (len(positions) == 3)}
                for num, positions in d_no_of_values.items():
                    positions = list(positions)
                    if (len(positions) == 2):
                        # check row
                        if (positions[0][0]) == (positions[1][0]):
                            # two possible locations for a given value in a box appear in the same row
                            update_row = positions[0][0]
                            for update_col in range(0, 9):
                                if (update_col != positions[0][1]) and (update_col != positions[1][1]):
                                    if num in self.get_possible_values(update_row, update_col):
                                         self.possible_values[update_row][update_col].remove(num)
                        # check col
                        if (positions[0][1]) == (positions[1][1]):
                            # two possible locations for a given value in a box appear in the same column
                            update_col = positions[0][1]
                            for update_row in range(0,9):
                                if (update_row != positions[0][0]) and (update_row != positions[1][0]):
                                    if num in self.get_possible_values(update_row, update_col):
                                        self.possible_values[update_row][update_col].remove(num)


                    if (len(positions) == 3):
                        # check row
                        if (positions[0][0] == positions[1][0] == positions[2][0]):
                            # two possible locations for a given value in a box appear in the same row
                            update_row = positions[0][0]
                            for update_col in range(0, 9):
                                if (update_col != positions[0][1]) and (update_col != positions[1][1]) and (update_col != positions[2][1]):
                                    if num in self.get_possible_values(update_row, update_col):
                                         self.possible_values[update_row][update_col].remove(num)
                        # check col
                        if (positions[0][1] == positions[1][1] == positions[2][1]):
                            # two possible locations  fora given value in a box appear in the same column
                            update_col = positions[0][1]
                            for update_row in range(0,9):
                                if (update_row != positions[0][0]) and (update_row != positions[1][0]) and (update_row != positions[2][0]):
                                    if num in self.get_possible_values(update_row, update_col):
                                        self.possible_values[update_row][update_col].remove(num)
                    
    # a constraint satisfaction that helps prune the number of possible game states
    def x_wing(self):

        # possible values
        for num in range(1, 10):
            value_rows = {}
            value_cols = {}
            for row in range(0,9):
                for column in range(0,9):
                    if num in self.get_possible_values(row, column):
                        if row in value_rows:
                            value_rows[row].add(column)
                        else:
                            value_rows[row] = set([column])
                        if column in value_cols:
                            value_cols[column].add(row)
                        else:
                            value_rows[column] = set([row])
    
            
            value_rows = {row: columns for row, columns in value_rows.items() if len(columns) == 2}
            value_cols = {column: rows for column, rows in value_cols.items() if len(rows) == 2}

            # rows
            doubles = combinations(value_rows.items(), 2)
            for double1, double2 in doubles:
                row1, columns1 = double1
                row2, columns2 = double2
                # check for x-wing quad
                if columns1 == columns2:
                    for update_col in columns1:
                        for update_row in range(0, 9):
                            if (update_row != row1) and (update_row != row2):
                                if num in self.get_possible_values(update_row,update_col):
                                    self.possible_values[update_row][update_col].remove(num)

            # columns
            doubles = combinations(value_cols.items(), 2)
            for double1, double2 in doubles:
                col1, rows1 = double1
                col2, rows2 = double2
                # check for x-wing quad
                if rows1 == rows2:
                    for update_row in rows1:
                        for update_col in range(0, 9):
                            if (update_col != col1) and (update_col != col2):
                                if num in self.get_possible_values(update_row,update_col):
                                    self.possible_values[update_row][update_col].remove(num)
    
    # a helper function for the AC3 algorithm
    def get_arc_queue(self):
        arc_queue = []

        for row in range(0, 9):
            positions = []
            for column in range(0, 9):
                positions.append((row, column))
            pairs_of_cells = combinations(positions, 2)
            for (Xi, Yi), (Xj, Yj) in pairs_of_cells:
                if Yi != Yj:
                    arc_queue.append(((Xi, Yi), (Xj, Yj)))

        for column in range(0, 9):
            positions = []
            for row in range(0, 9):
                positions.append((row, column))
            pairs_of_cells = combinations(positions, 2)
            for (Xi, Yi), (Xj, Yj) in pairs_of_cells:
                if Xi != Xj:
                    arc_queue.append(((Xi, Yi), (Xj, Yj)))

        for grid_row in range(0, 3):
            for grid_col in range(0,3):
                positions = []
                for row in range((grid_row * 3), ((grid_row +1) * 3)):
                    for column in range((grid_col * 3), ((grid_col + 1) * 3)):
                        positions.append((row, column))
                pairs_of_cells = combinations(positions, 2)
                for (Xi, Yi), (Xj, Yj) in pairs_of_cells:
                    if (Xi != Xj) and (Yi != Yj):
                        arc_queue.append(((Xi, Yi), (Xj, Yj)))

        return arc_queue

    # a helper function for the AC3 algorithm
    def get_neighbours(self, Xi, Yi):
        neighbours = []

        for column in range(0, 9):
            if column != Yi:
                neighbours.append((Xi, column))

        for row in range(0, 9):
            if row != Xi:
                neighbours.append((row, Yi))

        grid_row = Xi // 3
        grid_col = Yi // 3
        for row in range((grid_row * 3), ((grid_row + 1) * 3)):
            for column in range((grid_col * 3), ((grid_col + 1) * 3)):
                if (row != Xi) and (column != Yi):
                    neighbours.append((row, column))

        return neighbours
    
    # a helper function for the AC3 algorithm 
    def check_consistency(self, x, Xi, Yi, y, Xj, Yj):
        if (Xi == Xj) and (x == y):
            return False
        
        if (Yi == Yj) and (x ==y):
            return False
        
        if ((Xi // 3) == (Xj // 3)) and ((Yi // 3) == (Yj // 3)) and (x == y):
            return False
        
        return True

    # an algorithm used to determine arc-consistency between cells within a game state 
    def AC3(self):
        queue = self.get_arc_queue()
        while len(queue) > 0:
            ((Xi, Yi), (Xj, Yj)) = queue.pop()
            if self.revise(Xi, Yi, Xj, Yj):
                if len(self.get_possible_values(Xi, Yi)) == 0:
                    return False
                neighbours = self.get_neighbours(Xi, Yi)
                for (Xk, Yk) in neighbours:
                    if (Xk != Xj) and (Yk != Yj):
                        queue.append(((Xk, Yk),(Xi, Yi)))
        return True
    
    # a helper function for the AC3 algorithm 
    def revise(self,Xi, Yi, Xj, Yj):
        revised = False
        for x in self.get_possible_values(Xi, Yi):
            consistent = False
            for y in self.get_possible_values(Xj, Yj):
                if (self.check_consistency(x, Xi, Yi, y, Xj, Yj)):
                    consistent = True
            if not consistent:
                self.possible_values[Xi][Yi].remove(x)
                revised = True
            
        return revised

    # a function that returns the total number of possible moves remaining across the entire board
    def get_remaining_options(self):
        remaining_options = 0
        for row in self.possible_values:
            for possible_values in row:
                remaining_options += len(possible_values)
        return remaining_options

    # a function that returns the number of possible values a given cell on the board has remanining
    def get_possible_values(self, row, column):
        return self.possible_values[row][column].copy()

    # a function that returns all the cells that have only one possible value for assignment remaining 
    def get_singleton_columns(self):
        return [(row_index, column_index) for row_index, rows in enumerate(self.possible_values) 
                                          for column_index, possible_values in enumerate(rows)
                                          if len(possible_values) == 1 
                                          and self.final_values[row_index][column_index] == -1]

# a function for returning the value that causes the least amount of constraints on other variables (LCV)
def least_constraining_value(partial_state, row_index, col_index):
    values = partial_state.get_possible_values(row_index, col_index)
    new_states = [partial_state.set_value(row_index, col_index, value) for value in values]
    return (sorted(new_states, key = lambda new_state : new_state.get_remaining_options()))

# a function for returning the variable with the least amount of remaining values (MRV)
def minimum_remaining_values(partial_state):
    cell_indices = [(row_index, col_index) for row_index, rows in enumerate(partial_state.possible_values) 
                                            for col_index, possible_values in enumerate(rows)
                                            if len(possible_values) > 1]
    cell_indices =  (sorted(cell_indices, key = lambda cell_index : partial_state.get_possible_values(cell_index[0], cell_index[1]), reverse=True))
    return cell_indices[0]

# a function used to perform the backtracking depth-first search with constraint propagation
def backtracking_depth_first_search(partial_state):
       
    frontier = [partial_state]
    explored = []

    current_state = partial_state

    while not current_state.is_solved():

        explored.append(current_state)

        if current_state.is_invalid():
            frontier.pop()
        else:
            row_index, col_index = minimum_remaining_values(current_state)
            new_states = least_constraining_value(current_state, row_index, col_index)
            
            new_states_to_add_to_frontier = []
        
            for new_state in (new_states):

                if new_state.is_solved():
                    return new_state
        
                if (not new_state.is_invalid()):

                    new_state.naked_candidates()
                    new_state.intersection_removal()
                    count = new_state.get_remaining_options()
                    if count < 300:
                        new_state.hidden_candidates()
                    if count < 200:
                        new_state.x_wing()
                            
                    if (new_state not in explored) and (new_state not in frontier) and (not new_state.is_invalid()):
                        new_states_to_add_to_frontier.append(new_state)

            new_states_to_add_to_frontier.reverse()
            for new_state in new_states_to_add_to_frontier:
                frontier.append(new_state)

        if len(frontier) == 0:
            return None

        current_state = frontier.pop(-1)

    return current_state

# a function for running the sudoku solver
def sudoku_solver(sudoku):
    partial_state = PartialSudokuState(initial_config=sudoku.tolist())
    goal = backtracking_depth_first_search(partial_state)
    if goal == None:
        return np.array([[-1] * 9 for _ in range(0, 9)])
    return np.array(goal.final_values)

# a function for testing the sudoku solver on unseen sudoku's of varying difficulty 

SKIP_TESTS = True

if not SKIP_TESTS:
    import time
    import numpy as np
    __SCORES = {}
    difficulties = ['very_easy', 'easy', 'medium', 'hard']

    for difficulty in difficulties:
        print(f"Testing {difficulty} sudokus")
        
        sudokus = np.load(f"data/{difficulty}_puzzle.npy")
        solutions = np.load(f"data/{difficulty}_solution.npy")
        
        count = 0
        for i in range(len(sudokus)):
            sudoku = sudokus[i].copy()
            print(f"This is {difficulty} sudoku number", i)
            print(sudoku)
            
            start_time = time.process_time()
            your_solution = sudoku_solver(sudoku)
            end_time = time.process_time()
            
            if not isinstance(your_solution, np.ndarray):
                print("\033[91m[ERROR] Your sudoku_solver function returned a variable that has the incorrect type. If you submit this it will likely fail the auto-marking procedure result in a mark of 0 as it is expecting the function to return a numpy array with a shape (9,9).\n\t\033[94mYour function returns a {} object when {} was expected.\n\x1b[m".format(type(your_solution), np.ndarray))
            elif not np.all(your_solution.shape == (9, 9)):
                print("\033[91m[ERROR] Your sudoku_solver function returned an array that has the incorrect shape.  If you submit this it will likely fail the auto-marking procedure result in a mark of 0 as it is expecting the function to return a numpy array with a shape (9,9).\n\t\033[94mYour function returns an array with shape {} when {} was expected.\n\x1b[m".format(your_solution.shape, (9, 9)))
            
            print(f"This is your solution for {difficulty} sudoku number", i)
            print(your_solution)
            
            print("Is your solution correct?")
            if np.array_equal(your_solution, solutions[i]):
                print("Yes! Correct solution.")
                count += 1
            else:
                print("No, the correct solution is:")
                print(solutions[i])
            
            print("This sudoku took {} seconds to solve.\n".format(end_time-start_time))

        print(f"{count}/{len(sudokus)} {difficulty} sudokus correct")
        __SCORES[difficulty] = {
            'correct': count,
            'total': len(sudokus)
        }
