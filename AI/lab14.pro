% ===== Facts =====
thuc_an(ga).
thuc_an(tao).
song(bill).
an(bill, dau_phong).

% ===== Rules =====
thuc_an(X) :- an(Y, X), song(Y).   % Thứ mà ai đó ăn và vẫn còn sống cũng là thức ăn
an(john, X) :- thuc_an(X).         % John ăn tất cả những gì là thức ăn
an(sue, X)  :- an(bill, X).        % Sue ăn những thứ mà Bill ăn

% ===== Xuất kết quả =====
print_food :- forall(thuc_an(X), (write(X), nl)).
print_john :- forall(an(john, X), (write(X), nl)).
print_sue  :- forall(an(sue, X),  (write(X), nl)).
print_who  :- forall(an(A, B),    (write(A), write(' an '), write(B), nl)).

% ===== Main =====
main :-
    read_line_to_string(user_input, Cmd0),
    normalize_space(string(Cmd, Cmd0)),
    ( Cmd = "food" -> print_food
    ; Cmd = "john" -> print_john
    ; Cmd = "sue"  -> print_sue
    ; Cmd = "who"  -> print_who
    ; print_food
    ).

:- initialization(main).
