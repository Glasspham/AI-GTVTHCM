% solution.pro
% Sinh viên CHỈ sửa chỗ ???? để viết luật:
% "Mọi sinh viên ngành CNTT đều học môn TTNT"

:- initialization(main).
:- dynamic student/1, major/2, studies/2.

% ===== PHẦN SV CẦN HOÀN THÀNH =====
% Viết luật tổng quát:
studies(X, ttnt) :- student(X), major(X, cntt).

% ===== KHÔNG SỬA PHẦN DƯỚI =====
main :-
    load_kb,
    findall(S, studies(S, ttnt), L),
    sort(L, R),
    writeln(R),
    halt.

% Đọc facts từ stdin (VPL đưa Input= vào)
load_kb :-
    repeat,
        read(Term),
        ( Term == end_of_file -> !
        ; assertz(Term), fail).
