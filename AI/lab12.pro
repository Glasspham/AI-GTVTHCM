:- initialization(main).

main :-
    read_line_to_codes(user_input, Codes),
    ( Codes == end_of_file ->
        halt(1)
    ; atom_codes(Atom, Codes),
      atomic_list_concat(Atoms, ' ', Atom),
      maplist(atom_number, Atoms, Numbers),
      pair_jobs(Numbers, Jobs),
      print_jobs(Jobs),
      halt
    ).

pair_jobs([], []).
pair_jobs([ID, Duration | Rest], [[ID, Duration] | Jobs]) :-
    pair_jobs(Rest, Jobs).

print_jobs([]).
print_jobs([[ID, Duration] | Rest]) :-
    format('Job ~w: Start=0 End=~w~n', [ID, Duration]),
    print_jobs(Rest).
