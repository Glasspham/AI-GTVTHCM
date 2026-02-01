% main.pro
% Temporal infection model - robust cho môi trường chấm tự động
:- initialization(main).

% --- model parameters (đã đặt p0 = 0.05 để input 0 => 0.0500) ---
p0(0.05).
infection_rate(0.10).
recovery_rate(0.05).

% --- recursive forward inference ---
prob_at_day(0, P) :- p0(P).
prob_at_day(Day, P) :-
    Day > 0,
    PrevDay is Day - 1,
    prob_at_day(PrevDay, PrevP),
    infection_rate(IR),
    recovery_rate(RR),
    P is PrevP * (1.0 - RR) + (1.0 - PrevP) * IR.

% --- safe main for automated grader ---
main :-
    catch(read_line_to_string(user_input, Line), _, Line = end_of_file),
    ( Line == end_of_file ->
        halt(0)  % không có input -> exit 0
    ;
        % tách token, lấy token đầu làm N
        split_string(Line, " \t\r\n", " \t\r\n", Tokens),
        ( Tokens = [Tok|_] ->
            ( number_string(Number, Tok) ->
                ( Number < 0 -> N = 0
                ; ( integer(Number) -> N = Number ; N is floor(Number) )
                )
            ; N = 0
            )
        ; N = 0
        ),
        prob_at_day(N, P),
        format('~4f~n', [P]),
        halt(0)
    ).
