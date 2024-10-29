#include <bits/stdc++.h>
#define M_PI 3.14159265358979323846

using namespace std;

vector<string> split(const string& str, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(str);
    while (getline(tokenStream, token, delimiter)) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        tokens.push_back(token);
    }
    return tokens;
}

class NaiveBayes {
private:
    vector<int> classes;
    vector<double> priors;
    vector<vector<unordered_map<int, double>>> categorical_probs;
    vector<vector<pair<double, double>>> continuous_stats;
    vector<int> categorical_features;
    vector<int> continuous_features;
    int num_features;
    int num_classes;

public:
    NaiveBayes(const vector<int>& cat_features, const vector<int>& cont_features) 
        : categorical_features(cat_features), continuous_features(cont_features) {
        num_features = max(*max_element(cat_features.begin(), cat_features.end()),
                          *max_element(cont_features.begin(), cont_features.end())) + 1;
    }

    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        try {
            cout << "Starting fit method" << endl;
            cout << "Data size: " << X.size() << " samples, " << (X.empty() ? 0 : X[0].size()) << " features" << endl;
            
            if (X.empty() || y.empty() || X.size() != y.size()) {
                throw runtime_error("Invalid input dimensions");
            }

            unordered_map<int, int> class_counts;
            for (int label : y) {
                class_counts[label]++;
            }
            
            num_classes = class_counts.size();
            cout << "Found " << num_classes << " unique classes" << endl;
            for (const auto& pair : class_counts) {
                cout << "Class " << pair.first << ": " << pair.second << " instances" << endl;
            }
            
            classes.clear();
            priors.clear();
            classes.reserve(num_classes);
            priors.reserve(num_classes);
            categorical_probs.resize(num_classes, vector<unordered_map<int, double>>(num_features));
            continuous_stats.resize(num_classes, vector<pair<double, double>>(num_features));

            vector<vector<vector<double>>> class_feature_values(num_classes);
            for (int i = 0; i < num_classes; i++) {
                class_feature_values[i].resize(num_features);
            }

            // Calculate priors and collect feature values per class
            for (const auto& pair : class_counts) {
                int label = pair.first;
                int count = pair.second;
                classes.push_back(label);
                priors.push_back(static_cast<double>(count) / y.size());
            }

            // Collect feature values by class
            for (size_t i = 0; i < y.size(); ++i) {
                int class_idx = find(classes.begin(), classes.end(), y[i]) - classes.begin();
                for (int feat : continuous_features) {
                    if (feat < X[i].size()) {
                        class_feature_values[class_idx][feat].push_back(X[i][feat]);
                    }
                }
            }

            // Process each class
            for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
                cout << "Processing class " << class_idx + 1 << " of " << num_classes << endl;
                int current_class = classes[class_idx];

                // Process continuous features
                for (int feat : continuous_features) {
                    const auto& values = class_feature_values[class_idx][feat];
                    if (!values.empty()) {
                        double sum = accumulate(values.begin(), values.end(), 0.0);
                        double mean = sum / values.size();
                        double sq_sum = 0.0;
                        for (double val : values) {
                            sq_sum += val * val;
                        }
                        double variance = (sq_sum / values.size()) - (mean * mean);
                        variance = max(variance, 1e-10);
                        continuous_stats[class_idx][feat] = make_pair(mean, variance);
                    }
                }

                // Process categorical features
                for (int feat : categorical_features) {
                    unordered_map<int, int> value_counts;
                    int total_count = 0;
                    
                    for (size_t i = 0; i < y.size(); ++i) {
                        if (y[i] == current_class) {
                            value_counts[static_cast<int>(X[i][feat])]++;
                            total_count++;
                        }
                    }
                    
                    for (const auto& vc : value_counts) {
                        categorical_probs[class_idx][feat][vc.first] = 
                            static_cast<double>(vc.second) / total_count;
                    }
                }
            }
            
            cout << "Fit method completed successfully" << endl;
            
        } catch (const exception& e) {
            cerr << "Error in fit method: " << e.what() << endl;
            throw;
        }
    }

    inline double gaussian_pdf(double x, double mean, double variance) const {
        return exp(-0.5 * pow((x - mean) / sqrt(variance), 2)) / sqrt(2 * M_PI * variance);
    }

    double calculate_class_likelihood(const vector<double>& x, int class_idx) const {
        double log_likelihood = log(priors[class_idx]);

        for (int feat : continuous_features) {
            if (feat < x.size()) {
                double mean = continuous_stats[class_idx][feat].first;
                double var = continuous_stats[class_idx][feat].second;
                log_likelihood += log(gaussian_pdf(x[feat], mean, var));
            }
        }

        for (int feat : categorical_features) {
            if (feat < x.size()) {
                int value = static_cast<int>(x[feat]);
                const auto& prob_map = categorical_probs[class_idx][feat];
                auto it = prob_map.find(value);
                double prob = (it != prob_map.end()) ? it->second : 1e-10;
                log_likelihood += log(prob);
            }
        }

        return log_likelihood;
    }

    int predict(const vector<double>& x) const {
        int best_class = classes[0];
        double best_likelihood = calculate_class_likelihood(x, 0);
        
        for (int i = 1; i < num_classes; ++i) {
            double likelihood = calculate_class_likelihood(x, i);
            if (likelihood > best_likelihood) {
                best_likelihood = likelihood;
                best_class = classes[i];
            }
        }
        return best_class;
    }

    vector<int> predict_batch(const vector<vector<double>>& X) const {
        vector<int> predictions(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
};

void load_csv(const string& filename, vector<vector<double>>& X, vector<int>& y) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    string line;
    getline(file, line); // Skip header

    int count_leq_50k = 0;
    int count_gt_50k = 0;

    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        if (tokens.size() < 2) continue;
        
        vector<double> row;
        row.reserve(tokens.size() - 1);

        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            try {
                row.push_back(stod(tokens[i]));
            } catch (const invalid_argument&) {
                hash<string> hasher;
                row.push_back(static_cast<double>(hasher(tokens[i]) % 100000));
            }
        }
        
        string label = tokens.back();
        label.erase(0, label.find_first_not_of(" \t\r\n"));
        label.erase(label.find_last_not_of(" \t\r\n") + 1);
        
        if (label == "<=50K" || label == "<=50K.") {
            y.push_back(0);
            count_leq_50k++;
        } else if (label == ">50K" || label == ">50K.") {
            y.push_back(1);
            count_gt_50k++;
        } else {
            cerr << "Warning: Unknown class label: '" << label << "'" << endl;
            continue;
        }
        
        X.push_back(move(row));
    }

    cout << "Class distribution:" << endl;
    cout << "<=50K: " << count_leq_50k << " instances" << endl;
    cout << ">50K: " << count_gt_50k << " instances" << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    vector<vector<double>> X;
    vector<int> y;

    try {
        cout << "Loading data..." << endl;
        load_csv("adult.csv", X, y);
        cout << "Data loaded successfully" << endl;
        
        vector<int> categorical_features = {1, 3, 5, 6, 7, 8, 9, 13};
        vector<int> continuous_features = {0, 2, 4, 10, 11, 12};

        NaiveBayes model(categorical_features, continuous_features);
        cout << "Training model..." << endl;
        model.fit(X, y);
        cout << "Model trained successfully" << endl;

        vector<int> predictions = model.predict_batch(X);

        // Calculate metrics
        int correct = 0;
        int true_pos = 0, true_neg = 0, false_pos = 0, false_neg = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == y[i]) {
                correct++;
                if (y[i] == 1) true_pos++;
                else true_neg++;
            } else {
                if (predictions[i] == 1) false_pos++;
                else false_neg++;
            }
        }
        
        double accuracy = static_cast<double>(correct) / y.size();
        double precision = static_cast<double>(true_pos) / (true_pos + false_pos);
        double recall = static_cast<double>(true_pos) / (true_pos + false_neg);
        double f1_score = 2 * (precision * recall) / (precision + recall);
        
        cout << "\nModel Performance Metrics:" << endl;
        cout << "Accuracy: " << accuracy << endl;
        cout << "Precision: " << precision << endl;
        cout << "Recall: " << recall << endl;
        cout << "F1 Score: " << f1_score << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
