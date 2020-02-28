import pytest

from regex2DFA.infix_to_postfix import infix_to_postfix, Operator, Pairs, NonDeterministicFiniteAutomaton, concat, \
    merge_nfa, union, kleene_closure, postfix_to_nfa, NFAVisitor, nfa_to_dfa


# from regex2DFA.infix_to_postfix import *
# from regex2DFA.infix_to_postfix import _EPSILON

@pytest.mark.parametrize('regular_expression, expectation',
                         {
                             ('w.(w|d)*', 'wwd|*.'),
                             ('d.d*', 'dd*.'),
                             ('d.d*.w', 'dd*w..'),
                             ('4', '4')
                         }
                         )
def test_infix_to_postfix_regex(regular_expression: str, expectation: str):
    # id_regex = 'w.(w|d)*'
    # number_regex = 'd.d*'

    # operators = {'.': Operator('.', 0, 2),
    #              '|': Operator('|', 1, 2),
    #              '*': Operator('*', 2, 1)}

    operators = {'.': Operator('.', 0, 2),
                 '|': Operator('|', 1, 2),
                 '*': Operator('*', 2, 1)}

    pairs = Pairs({
        '(': ')',
        '[': ']'
    })

    # pairs = {'(': Pair('(', True, '(', ')'),
    #          ')': Pair(')', False, '(', ')')}

    assert infix_to_postfix(operators, pairs, regular_expression) == expectation


# def nfa_creation_from_epsilon(self):
#     expected_nfa = NonDeterministicFiniteAutomaton()
#
#     nfa = NonDeterministicFiniteAutomaton(_EPSILON_TRANSITION)
#
#     self.assertEqual(nfa, expected_nfa)

# class TestStateCreation:
#
#     def state_verification(self, state, is_end, transition, epsilon_transition):
#         assert state.is_end == is_end
#         assert state.transition == transition
#         assert state.epsilon_transition == epsilon_transition
#
#     def test_regular_state(self):
#         state = State()
#         self.state_verification(state, is_end=False, transition=tuple(), epsilon_transition=list())
#
#     def test_regular_state_explicit(self):
#         state = State(False)
#         self.state_verification(state, is_end=False, transition=tuple(), epsilon_transition=list())
#
#     def test_end_state(self):
#         state = State(True)
#         self.state_verification(state, is_end=True, transition=tuple(), epsilon_transition=list())
#
#     def test_add_epsilon_transition(self):
#         state = State()
#         target_uuid: UUID = uuid4()
#         state.add_epsilon_transition(target_uuid)
#         assert len(state.epsilon_transition) == 1
#         assert state.epsilon_transition[0] == target_uuid
#
#     def test_add_2_epsilon_transition(self):
#         state = State()
#         target_uuid_1: UUID = uuid4()
#         target_uuid_2: UUID = uuid4()
#         state.add_epsilon_transition(target_uuid_1)
#         state.add_epsilon_transition(target_uuid_2)
#         assert len(state.epsilon_transition) == 2
#         assert state.epsilon_transition[0] == target_uuid_1
#         assert state.epsilon_transition[1] == target_uuid_2
#
#     def test_add_3_epsilon_transition(self):
#         # should fail
#
#         state = State()
#         target_uuid_1: UUID = uuid4()
#         target_uuid_2: UUID = uuid4()
#         target_uuid_3: UUID = uuid4()
#         state.add_epsilon_transition(target_uuid_1)
#         state.add_epsilon_transition(target_uuid_2)
#         with pytest.raises(state.TooManyEpsilonTransition):
#             state.add_epsilon_transition(target_uuid_3)
#
#     def test_add_symbol_transition(self):
#         state = State()
#         target_uuid_1: UUID = uuid4()
#         symbol: str = 't'
#         state.add_transition(symbol, target_uuid_1)
#         assert state.transition == (target_uuid_1, symbol)
#
#     def test_add_2_symbol_transition(self):
#         state = State()
#         target_uuid_1: UUID = uuid4()
#         target_uuid_2: UUID = uuid4()
#         symbol_1: str = 't'
#         symbol_2: str = 'a'
#         state.add_transition(symbol_1, target_uuid_1)
#         with pytest.raises(state.TooManySymbolTransition):
#             state.add_transition(symbol_2, target_uuid_2)


# def test_concat(self):
#
#     expected_nfa = NonDeterministicFiniteAutomaton()
#
#     nfa_1 = NonDeterministicFiniteAutomaton()
#     nfa_2 = NonDeterministicFiniteAutomaton()
#     nfa_3 = concat(nfa_1, nfa_2)


class TestPairs:
    def test_pair_empty(self):
        pairs = Pairs()
        assert pairs.symbols == dict()

    def test_add_pair_from_empty(self):
        pairs = Pairs()
        pairs.add_pair('a', 'b')
        assert list(pairs.closers()) == ['b']
        assert pairs.get_opener('b') == 'a'
        assert set(pairs.openers()) == {'a'}
        assert set(pairs.get_symbols()) == {'a', 'b'}


class TestNFA:
    def test_nfa_creation(self):
        nfa = NonDeterministicFiniteAutomaton()
        assert nfa.transition_table[nfa.start_state] == dict()
        # assert nfa.transition_table[nfa.start_state]['epsilon'] == {nfa.end_state}

    def test_nfa_creation_symbol(self):
        nfa = NonDeterministicFiniteAutomaton('a')
        assert nfa.transition_table[nfa.start_state] == {'a': nfa.end_state}
        assert nfa.transition_table[nfa.start_state]['a'] == nfa.end_state

    def test_merge_nfa(self):
        nfa_1 = NonDeterministicFiniteAutomaton('a')
        nfa_2 = NonDeterministicFiniteAutomaton('b')
        nfa_3 = merge_nfa(nfa_1, nfa_2)

        assert nfa_3.start_state == nfa_1.start_state
        assert nfa_3.end_state == nfa_1.end_state
        assert nfa_3.start_state != nfa_2.start_state
        assert nfa_3.end_state != nfa_2.end_state
        assert nfa_3.get_transition(nfa_1.start_state, 'a') == nfa_1.end_state
        assert nfa_3.get_transition(nfa_2.start_state, 'b') == nfa_2.end_state

    def test_concat_nfa(self):
        nfa_1 = NonDeterministicFiniteAutomaton('a')
        nfa_2 = NonDeterministicFiniteAutomaton('b')
        nfa_3 = concat(nfa_1, nfa_2)

        assert nfa_3.start_state == nfa_1.start_state
        assert nfa_3.end_state == nfa_2.end_state
        end_of_1 = nfa_3.get_transition(nfa_3.start_state, 'a')
        assert end_of_1 == nfa_1.end_state
        start_of_2 = nfa_3.get_transition(end_of_1.pop(), 'epsilon')
        assert start_of_2 == {nfa_2.start_state}
        end_of_2 = nfa_3.get_transition(start_of_2.pop(), 'b')
        assert end_of_2 == nfa_2.end_state

    def test_union_nfa(self):
        nfa_1 = NonDeterministicFiniteAutomaton('a')
        nfa_2 = NonDeterministicFiniteAutomaton('b')
        nfa_3 = union(nfa_1, nfa_2)

        assert nfa_3.get_transition(nfa_3.start_state, 'epsilon') == {nfa_1.start_state, nfa_2.start_state}

        assert nfa_3.get_transition(nfa_1.start_state, 'a') == nfa_1.end_state
        assert nfa_3.get_transition(nfa_2.start_state, 'b') == nfa_2.end_state

        for e_end in nfa_1.end_state:
            assert nfa_3.get_transition(e_end, 'epsilon') == nfa_3.end_state
        for e_end in nfa_2.end_state:
            assert nfa_3.get_transition(e_end, 'epsilon') == nfa_3.end_state

    def test_kleene_closure(self):
        # nfa_1 = NonDeterministicFiniteAutomaton('a')
        # nfa_2 = kleene_closure(nfa_1)
        #
        # assert nfa_2.start_state != nfa_1.start_state
        # assert nfa_2.end_state != nfa_1.end_state
        #
        # assert nfa_2.get_transition(nfa_2.start_state, 'epsilon') == {nfa_1.start_state, nfa_2.end_state}
        # assert nfa_2.get_transition(nfa_1.start_state, 'a') == nfa_1.end_state
        # assert nfa_2.get_transition(nfa_1.end_state, 'epsilon') == {nfa_1.start_state, nfa_2.end_state}

        simple_nfa = NonDeterministicFiniteAutomaton('a')
        kleene_nfa = kleene_closure(simple_nfa)
        visitor = NFAVisitor(kleene_nfa)

        # follow
        visitor.follow_epsilon()
        assert simple_nfa.start_state in visitor.active_states

        # a, aa, aaa
        for n in range(1, 4):
            visitor.reset()
            for i in range(n):
                visitor.move('a')
            assert kleene_nfa.end_state.issubset(visitor.active_states)

    def test_too_many_epsilon_transition(self):
        nfa = NonDeterministicFiniteAutomaton('epsilon')
        new_state = nfa.add_state()
        nfa.add_epsilon_transition(nfa.start_state, new_state)
        new_state = nfa.add_state()
        try:
            nfa.add_epsilon_transition(nfa.start_state, new_state)
            assert False
        except nfa.TooManyEpsilonTransition:
            assert True

    def test_too_many_symbol_transition(self):
        nfa = NonDeterministicFiniteAutomaton('a')
        new_state = nfa.add_state()
        try:
            nfa.add_transition(nfa.start_state, 'a', new_state)
            assert False
        except NonDeterministicFiniteAutomaton.TooManySymbolTransition:
            assert True

    def test_add_symbol_transition(self):
        nfa = NonDeterministicFiniteAutomaton('a')
        new_state = nfa.add_state()
        nfa.add_transition(nfa.start_state, 'b', new_state)

        assert nfa.get_transition(nfa.start_state, 'b') == {new_state}


#
# class TestNFAVisitor:
#     def test_reset(self):
#         nfa = NonDeterministicFiniteAutomaton('a')


class TestPostfixToNFA:
    def test_empty_regex(self):
        postfix_regular_expression = ''
        nfa = postfix_to_nfa(postfix_regular_expression)
        assert nfa.get_transition(nfa.start_state, 'epsilon') == nfa.end_state

    def test_single_character(self):
        postfix_regular_expression = 'a'
        nfa = postfix_to_nfa(postfix_regular_expression)
        assert nfa.get_transition(nfa.start_state, 'a') == nfa.end_state

    # TODO
    # def test_bad_union(self):
    #     postfix_regular_expression = 'a|b'
    #     nfa = postfix_to_nfa(postfix_regular_expression)

    def test_union(self):
        postfix_regular_expression = 'ab|'
        nfa = postfix_to_nfa(postfix_regular_expression)

        # a route
        visitor = NFAVisitor(nfa)
        visitor.move('a')
        assert visitor.active_states == nfa.end_state

        # b route
        visitor.reset()
        visitor.move('b')
        assert visitor.active_states == nfa.end_state

    def test_concatenation(self):
        postfix_regular_expression = 'ab.'
        nfa = postfix_to_nfa(postfix_regular_expression)

        # a route
        visitor = NFAVisitor(nfa)
        visitor.move('a')
        visitor.move('b')
        assert visitor.active_states == nfa.end_state

    def test_kleene_closure(self):
        postfix_regular_expression = 'a*'
        nfa = postfix_to_nfa(postfix_regular_expression)

        # aa route
        visitor = NFAVisitor(nfa)
        visitor.move('a')
        assert nfa.end_state.issubset(visitor.active_states)

        # aaa route
        visitor.reset()
        visitor.move('a')
        visitor.move('a')
        assert nfa.end_state.issubset(visitor.active_states)

    def test_complex_expression(self):
        postfix_regular_expression = 'wwd|*.'
        nfa = postfix_to_nfa(postfix_regular_expression)

        visitor = NFAVisitor(nfa)
        visitor.move('w')
        visitor.move('w')
        visitor.move('w')
        visitor.move('d')
        visitor.move('w')
        visitor.move('d')
        visitor.move('w')

        assert nfa.end_state.issubset(visitor.active_states)


class TestNFAVisitor:
    def test_exiting_end_state(self):
        postfix_regular_expression = 'wwd|*.'
        nfa = postfix_to_nfa(postfix_regular_expression)

        visitor = NFAVisitor(nfa)
        visitor.move('w')
        visitor.move('a')

        assert not nfa.end_state.issubset(visitor.active_states)


class TestNfaToDFA:
    def test_nfa_to_dfa(self):
        nfa = postfix_to_nfa('wwd|*.')
        alphabet = {'w', 'd'}
        dfa = nfa_to_dfa(nfa, alphabet)

        # TODO assert...
        visitor = NFAVisitor(dfa)

        visitor.reset()
        visitor.move('w')
        visitor.move('d')
        visitor.move('w')
        assert visitor.active_states == dfa.end_state

        visitor.reset()
        visitor.move('d')
        assert visitor.active_states == {frozenset()}

