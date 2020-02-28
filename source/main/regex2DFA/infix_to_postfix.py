import copy
from typing import List, Dict, Tuple, Set, FrozenSet
from uuid import UUID, uuid4

_START = 0
_END = -1

alphabet = {
    'epsilon': 0,
    'w': 1,
    'd': 2}


# class OperatorSet:
#
#     def __init__(self):
#         self.operators: Dict[str, ] = dict()
#
class Operator:

    # properties
    # symbol: str
    # priority: int
    # arity: int

    def __init__(self, symbol, priority, arity):
        assert priority >= 0
        assert arity >= 0
        self.symbol = symbol
        self.priority = priority
        self.arity = arity  # arity is the number of arguments that a function or operator takes


class Pairs:
    # properties
    symbols: Dict[str, str]  # dictionary from closer to opener

    def __init__(self, symbol_dictionary=None):
        """

        :type symbol_dictionary: dict
        """

        if symbol_dictionary is None:
            symbol_dictionary = dict()

        self.symbols = {v: k for k, v in symbol_dictionary.items()}

    def add_pair(self, open_symbol: str, close_symbol: str):
        assert len(open_symbol) == 1
        assert len(close_symbol) == 1
        self.symbols[close_symbol] = open_symbol

    def get_opener(self, closer_symbol):
        return self.symbols[closer_symbol]

    def closers(self):
        return self.symbols.keys()

    def get_symbols(self):
        return list(self.symbols.keys()) + list(self.symbols.values())

    def openers(self):
        return self.symbols.values()


class FiniteAutomaton:
    def __init__(self, symbol: str = None):
        self.transition_table: Dict[UUID, Dict[str, Set[UUID]]] = dict()
        self.start_state: UUID = self.add_state()  # Add start state
        self.end_state: Set[UUID] = None
        if symbol:
            self._from_symbol(symbol)

    def _from_symbol(self, symbol: str):
        self.end_state = {self.add_state()}  # Add end state
        self.transition_table[self.start_state][symbol] = self.end_state

    def get_transition(self, state, symbol):
        if symbol in self.transition_table[state]:
            return self.transition_table[state][symbol]
        else:
            return {}

    def add_state(self):
        uuid = uuid4()
        self.transition_table[uuid] = dict()
        return uuid

    def add_transition(self, source: UUID, symbol: str, target: UUID):
        if symbol not in self.transition_table[source].keys():
            self.transition_table[source][symbol] = {target}
        else:
            raise self.TooManySymbolTransition(symbol)

    class TooManyTransition(Exception):
        pass

    class TooManySymbolTransition(TooManyTransition):
        def __init__(self, symbol: str):
            Exception.__init__(self, f'State already has a symbol transition. Symbol: " {symbol} " can not be added.')
            # self.uuid = uuid


class NonDeterministicFiniteAutomaton(FiniteAutomaton):

    def add_epsilon_transition(self, source: UUID, target: UUID):
        if 'epsilon' not in self.transition_table[source].keys():
            self.transition_table[source]['epsilon'] = {target}
        elif len(self.transition_table[source]['epsilon']) < 2:
            self.transition_table[source]['epsilon'].add(target)
        else:
            raise self.TooManyEpsilonTransition(source)

    class TooManyEpsilonTransition(FiniteAutomaton.TooManyTransition):
        def __init__(self, uuid: UUID):
            Exception.__init__(self, f'State already has 2 epsilon transitions. Another can not be added.')
            self.uuid = uuid


class NFAVisitor:

    def __init__(self, nfa: NonDeterministicFiniteAutomaton):
        self.nfa = nfa
        self.active_states: Set[UUID] = set()
        self.reset()

    def reset(self):
        self.active_states = {self.nfa.start_state}

    def move(self, symbol: str):
        self.follow_epsilon()
        new_states = set()
        for e_state in self.active_states:
            new_states.update(self.nfa.get_transition(e_state, symbol))
        self.active_states = new_states
        self.follow_epsilon()

    def follow_epsilon(self):
        state_stack = self.active_states
        final_states = set()
        while len(state_stack):
            current_state = state_stack.pop()
            new_states = self.nfa.get_transition(current_state, 'epsilon')
            if new_states == {}:
                final_states.add(current_state)
            else:
                state_stack = state_stack.union(new_states)
        self.active_states = final_states


class DeterministicFiniteAutomaton(FiniteAutomaton):
    pass
    # def __init__(self):
    #     super().__init__()


def infix_to_postfix(operators: dict, pairs: Pairs, regex: str):
    stack = []
    answer = ''
    for e_character in regex:
        if e_character in operators.keys():
            while (len(stack) > 0 and
                   stack[-1] not in pairs.openers() and
                   operators[e_character].priority < operators[
                       stack[-1]].priority):
                answer += stack.pop()
            stack.append(e_character)
        elif e_character in pairs.openers():
            stack.append(e_character)
        elif e_character in pairs.closers():
            while stack[-1] not in pairs.get_opener(e_character):
                answer += stack.pop()
            stack.pop()
        else:
            answer += e_character

    for i in range(len(stack)):
        answer += stack.pop()

    return answer


def kleene_closure(nfa: NonDeterministicFiniteAutomaton):
    # TODO: how does kleene closure react to multiple ends
    # repeat *

    closed_nfa = copy_nfa(nfa)

    # create new start
    new_start = closed_nfa.add_state()
    # create new end
    new_end = closed_nfa.add_state()
    # add epsilon transition from new start to new end
    closed_nfa.add_epsilon_transition(new_start, new_end)
    # add epsilon transition from new start to old start
    closed_nfa.add_epsilon_transition(new_start, nfa.start_state)
    # add epsilon transition from all old ends to new end
    for e_end in nfa.end_state:
        closed_nfa.add_epsilon_transition(e_end, new_end)
    # add epsilon transition from old ends to old start
    for e_end in nfa.end_state:
        closed_nfa.add_epsilon_transition(e_end, nfa.start_state)
    # set nfa start to new start
    closed_nfa.start_state = new_start
    # set nfa end to new end
    closed_nfa.end_state = {new_end}

    return closed_nfa


def union(first: NonDeterministicFiniteAutomaton, second: NonDeterministicFiniteAutomaton):
    # or |

    unioned_nfa = merge_nfa(first, second)

    # create new start
    new_start = unioned_nfa.add_state()
    unioned_nfa.start_state = new_start

    # add epsilon from new start to nfa 1
    unioned_nfa.add_epsilon_transition(new_start, first.start_state)

    # add epsilon from new start to nfa 2
    unioned_nfa.add_epsilon_transition(new_start, second.start_state)

    # create new end
    new_end = unioned_nfa.add_state()
    unioned_nfa.end_state = {new_end}

    # add epsilon from all ends of nfa1 to new end
    for e_end in first.end_state:
        unioned_nfa.add_epsilon_transition(e_end, new_end)

    # add epsilon from all ends of nfa2 to new end
    for e_end in second.end_state:
        unioned_nfa.add_epsilon_transition(e_end, new_end)

    return unioned_nfa


def concat(first: NonDeterministicFiniteAutomaton, second: NonDeterministicFiniteAutomaton):
    """
    equivalent to and
    equivalent to .
    takes 2 NFAs
    adds epsilon transition from end of first NFA to start of second NFA
    set is_end end State of first NFA
    :param first:
    :param second:
    :return:
    """

    concat_nfa = merge_nfa(first, second)

    # adds epsilon transition from all ends of first NFA to start of second NFA
    for end_state in first.end_state:
        concat_nfa.transition_table[end_state]['epsilon'] = {second.start_state}

    # set is_end end State of first NFA
    concat_nfa.end_state = second.end_state

    return concat_nfa


def merge_nfa(first, second):
    """
    merges 2 nfas keeps start and end of the first one
    :param first:
    :param second:
    :return:
    """

    merged_nfa = copy.deepcopy(first)
    merged_nfa.transition_table = {**first.transition_table, **second.transition_table}

    return merged_nfa


def copy_nfa(nfa: NonDeterministicFiniteAutomaton):
    """

    :param nfa:
    :return:
    """
    return copy.deepcopy(nfa)


def postfix_to_nfa(regular_expression: str) -> NonDeterministicFiniteAutomaton:
    """

    :param regular_expression:
    :return:
    """
    if regular_expression == '':
        return NonDeterministicFiniteAutomaton('epsilon')

    stack: List[NonDeterministicFiniteAutomaton] = list()
    for e_character in regular_expression:
        if e_character == '.':
            right = stack.pop()
            left = stack.pop()
            stack.append(concat(left, right))
        elif e_character == '*':
            stack.append(kleene_closure(stack.pop()))
        elif e_character == '|':
            right = stack.pop()
            left = stack.pop()
            stack.append(union(left, right))
        else:
            stack.append(NonDeterministicFiniteAutomaton(e_character))

    return stack.pop()


def nfa_to_dfa(nfa: NonDeterministicFiniteAutomaton, alphabet: Set[str]):
    """

    :type alphabet:
    :param nfa:
    :param alphabet:
    :return:
    """
    # implements a depth first search
    nfa_visited_states: Set[FrozenSet[UUID]] = set()

    nfa_visitor: NFAVisitor = NFAVisitor(nfa)
    nfa_visitor.follow_epsilon()

    state_stack: List[FrozenSet[UUID]] = [frozenset(nfa_visitor.active_states)]

    simplified_nfa_transition_table: Dict[FrozenSet[UUID], Dict[str, FrozenSet[UUID]]] = {}

    while state_stack:
        vertex = state_stack.pop()
        if vertex not in nfa_visited_states:
            nfa_visited_states.add(vertex)

            # dfa_vertex = uuid4()
            simplified_nfa_transition_table[vertex] = dict()
            # dfa_states.add(dfa_vertex)
            # nfa_to_dfa_states[vertex] = dfa_vertex

            for letter in alphabet:
                nfa_visitor.active_states = set(vertex)
                nfa_visitor.move(letter)
                active_states = frozenset(nfa_visitor.active_states)
                state_stack.append(active_states)
                simplified_nfa_transition_table[vertex][letter] = active_states

    ###
    dfa_states = dict()
    for vertex in simplified_nfa_transition_table:
        if vertex:
            dfa_states[vertex] = uuid4()
        else:
            dfa_states[vertex] = frozenset()

    dfa = DeterministicFiniteAutomaton()
    dfa.end_state = set()

    dfa_transition_table = dict()
    for vertex in simplified_nfa_transition_table:
        dfa_vertex = dfa_states[vertex]
        dfa_transition_table[dfa_vertex] = dict()

        # find starts
        if nfa.start_state in vertex:
            dfa.start_state = dfa_vertex

        # find ends
        if not nfa.end_state.isdisjoint(vertex):
            dfa.end_state.add(dfa_vertex)

        for letter in alphabet:
            dfa_transition_table[dfa_vertex][letter] = {dfa_states[simplified_nfa_transition_table[vertex][letter]]}



    # replace visited_states in the transition table with individual UUID
    # for e_state in dfa_transition_table:
    #     if e_state == nfa.start_state:
    #         dfa.start_state = dfa_transition_table
    #     for letter in alphabet:
    #         dfa_transition_table[e_state][letter] = nfa_to_dfa_states[dfa_transition_table[e_state][letter]]

    dfa.transition_table = dfa_transition_table
    return dfa


# def clean_dfa():


# dfa: DeterministicFiniteAutomaton = DeterministicFiniteAutomaton()
# dfa.start_state = nfa.start_state
# for state in dfa_transition_table:
#     if nfa.end_state
#     dfa.add_transition()
#     for letter in state:
#
# return dfa


if __name__ == '__main__':
    # dd = FiniteAutomaton()
    # print('ehhlo')
    id_regex = 'w.(w|d)*'
    number_regex = 'd.d*'

    operators = {'.': Operator('.', 0, 2),
                 '|': Operator('|', 1, 2),
                 '*': Operator('*', 2, 1)}

    pairs = Pairs()
    pairs.add_pair('(', ')')

    postfix_id_regex = infix_to_postfix(operators, pairs, id_regex)
    # print(infix_to_postfix(operators, pairs, number_regex))

    nfa = postfix_to_nfa(postfix_id_regex)
    dfa = nfa_to_dfa(nfa, {'w', 'd'})

    #
    test_string = 'ww wd wwwddd wdwwwdwd dwwww wwd'
    # test_string = list(id_string)
    visitor = NFAVisitor(dfa)
    accumulator = ''
    test_string = list(test_string)
    while test_string:
        character = test_string.pop(0)
        visitor.move(character)
        if visitor.active_states:
            accumulator += character
        else:
            print(accumulator)

            print(f'Bad character: "{character}"')
            accumulator = ''
            visitor.reset()


