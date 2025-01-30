% Step 1: Load the FASTA files for donor and acceptor splice sites
Donor_Train_Positive = fastaread('Donor_Train_Positive.fasta');
Donor_Train_Negative = fastaread('Donor_Train_Negative.fasta');
Donor_Test_Positive = fastaread('Donor_Test_Positive.fasta');
Donor_Test_Negative = fastaread('Donor_Test_Negative.fasta');

Acceptor_Train_Positive = fastaread('Acceptor_Train_Positive.fasta');
Acceptor_Train_Negative = fastaread('Acceptor_Train_Negative.fasta');
Acceptor_Test_Positive = fastaread('Acceptor_Test_Positive.fasta');
Acceptor_Test_Negative = fastaread('Acceptor_Test_Negative.fasta');

% Step 2: Convert DNA sequence into CGR coordinates
function cgr_coords = sequence_to_cgr(sequence)
    % Initialize empty coordinates matrix (two dimensions for each nucleotide)
    cgr_coords = zeros(length(sequence), 2);
    
    % Mapping nucleotides to CGR coordinates (A, T, C, G -> 4 directions)
    for i = 1:length(sequence)
        switch sequence(i)
            case 'A'
                cgr_coords(i, :) = [1, 0];
            case 'T'
                cgr_coords(i, :) = [0, 1];
            case 'C'
                cgr_coords(i, :) = [1, 1];
            case 'G'
                cgr_coords(i, :) = [0, 0];
        end
    end
end

% Step 3: Prepare the donor training dataset
X_donor_train = [];
y_donor_train = [];

for i = 1:length(Donor_Train_Positive)
    cgr_coords = sequence_to_cgr(Donor_Train_Positive(i).Sequence);
    X_donor_train = [X_donor_train; cgr_coords(:)']; % Flatten to row vector
    y_donor_train = [y_donor_train; 1];
end

for i = 1:length(Donor_Train_Negative)
    cgr_coords = sequence_to_cgr(Donor_Train_Negative(i).Sequence);
    X_donor_train = [X_donor_train; cgr_coords(:)'];
    y_donor_train = [y_donor_train; 0];
end

assert(size(X_donor_train, 1) == length(y_donor_train), 'Mismatch in donor training data.');

% Step 4: Prepare the donor test dataset
X_donor_test = [];
y_donor_test = [];

for i = 1:length(Donor_Test_Positive)
    cgr_coords = sequence_to_cgr(Donor_Test_Positive(i).Sequence);
    X_donor_test = [X_donor_test; cgr_coords(:)'];
    y_donor_test = [y_donor_test; 1];
end

for i = 1:length(Donor_Test_Negative)
    cgr_coords = sequence_to_cgr(Donor_Test_Negative(i).Sequence);
    X_donor_test = [X_donor_test; cgr_coords(:)'];
    y_donor_test = [y_donor_test; 0];
end

% Step 5: Prepare the acceptor training dataset
X_acceptor_train = [];
y_acceptor_train = [];

for i = 1:length(Acceptor_Train_Positive)
    cgr_coords = sequence_to_cgr(Acceptor_Train_Positive(i).Sequence);
    X_acceptor_train = [X_acceptor_train; cgr_coords(:)'];
    y_acceptor_train = [y_acceptor_train; 1];
end

for i = 1:length(Acceptor_Train_Negative)
    cgr_coords = sequence_to_cgr(Acceptor_Train_Negative(i).Sequence);
    X_acceptor_train = [X_acceptor_train; cgr_coords(:)'];
    y_acceptor_train = [y_acceptor_train; 0];
end

assert(size(X_acceptor_train, 1) == length(y_acceptor_train), 'Mismatch in acceptor training data.');

% Step 6: Prepare the acceptor test dataset
X_acceptor_test = [];
y_acceptor_test = [];

for i = 1:length(Acceptor_Test_Positive)
    cgr_coords = sequence_to_cgr(Acceptor_Test_Positive(i).Sequence);
    X_acceptor_test = [X_acceptor_test; cgr_coords(:)'];
    y_acceptor_test = [y_acceptor_test; 1];
end

for i = 1:length(Acceptor_Test_Negative)
    cgr_coords = sequence_to_cgr(Acceptor_Test_Negative(i).Sequence);
    X_acceptor_test = [X_acceptor_test; cgr_coords(:)'];
    y_acceptor_test = [y_acceptor_test; 0];
end

% Step 7: Train the donor neural network
donor_net = feedforwardnet(10);
donor_net = train(donor_net, X_donor_train', y_donor_train');

% Step 8: Train the acceptor neural network
acceptor_net = feedforwardnet(20);
acceptor_net = train(acceptor_net, X_acceptor_train', y_acceptor_train');

% Step 9: Evaluate donor and acceptor models
y_donor_pred = donor_net(X_donor_test')';
y_acceptor_pred = acceptor_net(X_acceptor_test')';

% Step 10: Plot ROC for donor
[donor_fpr, donor_tpr, donor_thresholds] = perfcurve(y_donor_test, y_donor_pred, 1);
figure;
plot(donor_fpr, donor_tpr, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Donor ROC Curve');
grid on;

% Step 11: Plot ROC for acceptor
[acceptor_fpr, acceptor_tpr, acceptor_thresholds] = perfcurve(y_acceptor_test, y_acceptor_pred, 1);
figure;
plot(acceptor_fpr, acceptor_tpr, 'r-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Acceptor ROC Curve');
grid on;

% Step 12: Display AUC
donor_auc = trapz(donor_fpr, donor_tpr);
acceptor_auc = trapz(acceptor_fpr, acceptor_tpr);
disp(['Donor AUC: ', num2str(donor_auc)]);
disp(['Acceptor AUC: ', num2str(acceptor_auc)]);