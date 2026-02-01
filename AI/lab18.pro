% Rain–Umbrella — SV điền ??? (4 chỗ)

prior(0.5).   % <--- chỗ ??? số 1

p_rain_given_rain(0.7).
p_rain_given_norain(0.3).
p_u_given_rain(0.9).
p_u_given_norain(0.2).

obs_likelihood(u, LR, LNR) :- p_u_given_rain(LR), p_u_given_norain(LNR).
obs_likelihood(n, LR, LNR) :- p_u_given_rain(UR), p_u_given_norain(UNR), LR is 1.0-UR, LNR is 1.0-UNR.

predict(PPrev, Pred) :-
    p_rain_given_rain(A), p_rain_given_norain(B),
    Pred is A * PPrev + B * (1.0 - PPrev).   % <--- chỗ ??? số 2

update(Pred, Obs, PPost) :-
    obs_likelihood(Obs, LR, LNR),
    Numer is LR * Pred,                      % <--- chỗ ??? số 3
    Denom is LR * Pred + LNR * (1.0 - Pred), % <--- chỗ ??? số 4
    PPost is Numer / Denom.

forward([], P, P).
forward([O|T], P0, Pn) :-
    predict(P0, Pred),
    update(Pred, O, P1),
    forward(T, P1, Pn).

filter_obs([], []).
filter_obs([H|T], R) :-
    ( H='u' -> R=[u|R1]
    ; H='n' -> R=[n|R1]
    ; R=R1
    ), filter_obs(T, R1).

:- initialization(main).
main :-
    ( read_line_to_string(user_input, S0) -> true ; S0 = "" ),
    string_chars(S0, Cs),
    filter_obs(Cs, Obs),
    prior(P0), forward(Obs, P0, P),
    format('~4f~n', [P]), halt.
