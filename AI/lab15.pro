% main.pro
% Bayes đơn giản - robust cho môi trường chấm tự động

:- initialization(main).

% ==== KB ====
p_flu(0.1).
p_symptom_given_flu(cough, 0.8).
p_symptom_given_flu(fever, 0.7).
p_symptom_given_notflu(cough, 0.3).
p_symptom_given_notflu(fever, 0.1).

% ==== likelihood helpers: nếu không biết triệu chứng thì coi là trung tính (1.0)
likelihood_flu(S, P) :-
    (   p_symptom_given_flu(S, P) -> true ; P = 1.0 ).

likelihood_notflu(S, P) :-
    (   p_symptom_given_notflu(S, P) -> true ; P = 1.0 ).

% ==== multiply likelihoods
product_likelihoods_flu([], 1.0).
product_likelihoods_flu([S|Rest], P) :-
    likelihood_flu(S, PS),
    product_likelihoods_flu(Rest, PR),
    P is PS * PR.

product_likelihoods_notflu([], 1.0).
product_likelihoods_notflu([S|Rest], P) :-
    likelihood_notflu(S, PS),
    product_likelihoods_notflu(Rest, PR),
    P is PS * PR.

% ==== posterior
posterior_flu(Symptoms, P) :-
    p_flu(PF),
    product_likelihoods_flu(Symptoms, LF),
    product_likelihoods_notflu(Symptoms, LNF),
    N is PF * LF,
    D is N + (1.0 - PF) * LNF,
    ( D =:= 0.0 -> P = 0.0 ; P is N / D ).

% ==== string/atom helpers (loại bỏ chuỗi rỗng, chuyển về chữ thường)
filter_nonempty([], []).
filter_nonempty([""|T], R) :- !, filter_nonempty(T, R).
filter_nonempty([H|T], [H|R]) :- filter_nonempty(T, R).

strings_to_atoms_lower([], []).
strings_to_atoms_lower([S|Ts], [A|As]) :-
    string_lower(S, SL),
    atom_string(A, SL),
    strings_to_atoms_lower(Ts, As).

% ==== main: đọc 1 dòng từ stdin, an toàn với EOF và dòng rỗng
main :-
    catch(read_line_to_string(user_input, Str), _, Str = end_of_file),
    ( Str == end_of_file ->
        % không có input -> thoát bình thường (không in gì)
        halt(0)
    ;  % tách token theo khoảng trắng, loại bỏ token rỗng
       split_string(Str, " ", " \t\r\n", Tokens0),
       filter_nonempty(Tokens0, Tokens1),
       ( Tokens1 == [] -> Symptoms = [] ; strings_to_atoms_lower(Tokens1, Symptoms) ),
       posterior_flu(Symptoms, P),
       format('~4f~n', [P]),
       halt(0)
    ).
