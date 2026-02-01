% main.pro
% Chương trình chạy tự động cho tool chấm.
% Input: một dòng chứa "positive" hoặc "negative" (không phân biệt hoa/thường).
% Output: một số thực (4 chữ số thập phân), rồi exit.

:- initialization(main).

% Thông số bài toán
p_disease(0.005).       % prevalence
sensitivity(0.99).      % Se
specificity(0.95).      % Sp

% Tính hậu nghiệm khi test dương
posterior_disease(positive, P) :-
    p_disease(PD), sensitivity(Se), specificity(Sp),
    N is Se * PD,
    D is Se * PD + (1.0 - Sp) * (1.0 - PD),
    ( D =:= 0.0 -> P = 0.0 ; P is N / D ).

% Tính hậu nghiệm khi test âm
posterior_disease(negative, P) :-
    p_disease(PD), sensitivity(Se), specificity(Sp),
    N is (1.0 - Se) * PD,
    D is (1.0 - Se) * PD + Sp * (1.0 - PD),
    ( D =:= 0.0 -> P = 0.0 ; P is N / D ).

% Hàm chính: đọc 1 dòng từ stdin, an toàn với EOF và dòng rỗng
main :-
    catch(
        ( read_line_to_string(user_input, Str),
          ( Str == end_of_file ->
              % không có input -> thoát yên
              halt(0)
          ;
              % tách token, lấy token đầu, chuyển thành chữ thường
              split_string(Str, " \t\r\n", " \t\r\n", Tokens),
              ( Tokens = [T|_] ->
                  string_lower(T, TL),
                  ( (TL = "positive" ; TL = "+") -> Key = positive
                  ; (TL = "negative" ; TL = "-") -> Key = negative
                  ; % token không hợp lệ -> thoát (không in gì)
                    halt(0)
                  ),
                  posterior_disease(Key, P),
                  format('~4f~n', [P]),
                  halt(0)
              ; % không có token -> thoát
                halt(0)
              )
          )
        ),
        _Err,
        % nếu có exception (ví dụ môi trường khác), thoát êm
        halt(0)
    ).
