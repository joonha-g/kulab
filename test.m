% ================================
% MP3 파일 로드 및 파형/벡터 추출 코드
% ================================

clc; clear; close all;

% 1. MP3 파일 불러오기
[signal, Fs] = audioread('C:\Users\als15\Downloads\Quick Share\Over_the_Horizon.mp3');
% signal: 샘플 데이터 벡터, Fs: 샘플링 주파수

% 2. 시간축 생성
t = (0:length(signal)-1) / Fs;           % 초 단위 시간 벡터

% 3. 스테레오 채널 분리
if size(signal,2) == 2
    left = signal(:,1);
    right = signal(:,2);
else
    left = signal;
    right = [];
end

% 4. 파형 시각화
figure;
subplot(2,1,1);
plot(t, left);
xlabel('Time (s)'); ylabel('Amplitude');
title('Left Channel');
grid on;

if ~isempty(right)
    subplot(2,1,2);
    plot(t, right);
    xlabel('Time (s)'); ylabel('Amplitude');
    title('Right Channel');
    grid on;
end

% 5. 데이터 확인
disp('샘플링 주파수 (Hz):');
disp(Fs);
disp('데이터 크기 (행 x 열):');
disp(size(signal));
disp('앞 10개의 벡터값:');
disp(signal(1:10,:));

% 6. 시간축 포함한 데이터 생성
if ~isempty(right)
    saveData = [t(:), left(:), right(:)];
else
    saveData = [t(:), left(:)];
end

% 7. 텍스트 파일로 저장 (time, left, right)
save('wave_vector.txt', 'saveData', '-ascii');

% 8. 저장 완료 메시지
disp('wave_vector.txt 파일로 저장 완료되었습니다.');
disp('각 행은 [time, left, right] 형식입니다.');
