% ==========================================
% wave_vector.txt → 오디오 복원 및 재생 코드
% ==========================================

clc; clear; close all;

% 1. 텍스트 파일 불러오기
data = load('wave_vector.txt');   % time, left, right 형식

% 2. 열 개수 확인
[nRows, nCols] = size(data);
fprintf('불러온 데이터 크기: %d행 × %d열\n', nRows, nCols);

% 3. 채널 분리
if nCols == 3
    t = data(:,1);
    left = data(:,2);
    right = data(:,3);
    signal = [left, right];
else
    t = data(:,1);
    left = data(:,2);
    signal = left;  % 모노일 경우
end 

% 4. 샘플링 주파수 복원
% 시간 간격이 일정하다고 가정하여 Fs 계산
dt = mean(diff(t));     % 평균 시간 간격
Fs = round(1/dt);       % 샘플링 주파수 근사 복원
fprintf('복원된 샘플링 주파수: %.1f Hz\n', Fs);

% 5. 오디오 재생
p = audioplayer(signal, Fs);
play(p);

% 6. 파형 확인
figure;
if size(signal,2) == 2
    subplot(2,1,1);
    plot(t, signal(:,1)); title('Left Channel'); xlabel('Time (s)'); ylabel('Amplitude');
    subplot(2,1,2);
    plot(t, signal(:,2)); title('Right Channel'); xlabel('Time (s)'); ylabel('Amplitude');
else
    plot(t, signal); title('Mono Signal'); xlabel('Time (s)'); ylabel('Amplitude');
end

% 7. (선택) wav 파일로 저장
audiowrite('reconstructed.wav', signal, Fs);
disp('reconstructed.wav 파일로 저장 완료!');
