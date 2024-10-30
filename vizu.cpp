#include <bits/stdc++.h>
using namespace std;

struct Person {
    int age;
    std::string workclass;
    int fnlwgt;
    std::string education;
    int education_num;
    std::string marital_status;
    std::string occupation;
    std::string relationship;
    std::string race;
    std::string gender;
    int capital_gain;
    int capital_loss;
    int hours_per_week;
    std::string native_country;
    std::string income;
};

class CorrelationAnalyzer {
private:
    double calculateMean(const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }

    double calculateStandardDeviation(const std::vector<double>& values, double mean) {
        double sum_squares = 0.0;
        for (double value : values) {
            sum_squares += (value - mean) * (value - mean);
        }
        return std::sqrt(sum_squares / (values.size() - 1));
    }

public:
    double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            return 0.0;
        }

        double x_mean = calculateMean(x);
        double y_mean = calculateMean(y);
        double x_std = calculateStandardDeviation(x, x_mean);
        double y_std = calculateStandardDeviation(y, y_mean);

        if (x_std == 0.0 || y_std == 0.0) {
            return 0.0;
        }

        double sum_xy = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            sum_xy += (x[i] - x_mean) * (y[i] - y_mean);
        }

        return sum_xy / ((x.size() - 1) * x_std * y_std);
    }
};

class DataAnalyzer {
private:
    struct Feature {
        std::string name;
        std::vector<double> values;
    };
    std::vector<Feature> numerical_features;
    
    void initializeFeatures() {
        numerical_features = {
            {"Age", {}},
            {"Education_Num", {}},
            {"Capital_Gain", {}},
            {"Capital_Loss", {}},
            {"Hours_Per_Week", {}},
            {"Income_Binary", {}} // 0 for <=50K, 1 for >50K
        };
    }

public:
    void processData(const std::vector<Person>& data) {
        initializeFeatures();
        
        for (const auto& person : data) {
            numerical_features[0].values.push_back(person.age);
            numerical_features[1].values.push_back(person.education_num);
            numerical_features[2].values.push_back(person.capital_gain);
            numerical_features[3].values.push_back(person.capital_loss);
            numerical_features[4].values.push_back(person.hours_per_week);
            numerical_features[5].values.push_back(person.income.find(">50K") != std::string::npos ? 1.0 : 0.0);
        }
    }

    void generateCorrelationMatrix(const std::string& filename) {
        CorrelationAnalyzer correlator;
        int n = numerical_features.size();
        std::vector<std::vector<double>> correlation_matrix(n, std::vector<double>(n));

        // Calculate correlation matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                correlation_matrix[i][j] = correlator.calculateCorrelation(
                    numerical_features[i].values,
                    numerical_features[j].values
                );
            }
        }

        // Write correlation matrix to CSV
        std::ofstream outFile(filename);
        outFile << "Feature";
        for (const auto& feature : numerical_features) {
            outFile << "," << feature.name;
        }
        outFile << "\n";

        for (int i = 0; i < n; i++) {
            outFile << numerical_features[i].name;
            for (int j = 0; j < n; j++) {
                outFile << "," << std::fixed << std::setprecision(3) << correlation_matrix[i][j];
            }
            outFile << "\n";
        }
        outFile.close();

        // Generate ASCII visualization of correlation matrix
        std::ofstream vizFile(filename + ".txt");
        vizFile << "Correlation Matrix Visualization\n\n";
        
        // Print header
        vizFile << std::setw(15) << " ";
        for (int i = 0; i < n; i++) {
            vizFile << std::setw(15) << numerical_features[i].name.substr(0, 12);
        }
        vizFile << "\n";

        // Print matrix with ASCII characters
        for (int i = 0; i < n; i++) {
            vizFile << std::setw(15) << numerical_features[i].name.substr(0, 12);
            for (int j = 0; j < n; j++) {
                double corr = correlation_matrix[i][j];
                std::string symbol;
                if (corr == 1.0) symbol = "█";
                else if (corr > 0.7) symbol = "▓";
                else if (corr > 0.4) symbol = "▒";
                else if (corr > 0.2) symbol = "░";
                else if (corr > -0.2) symbol = " ";
                else if (corr > -0.4) symbol = "╳";
                else if (corr > -0.7) symbol = "╳╳";
                else symbol = "╳╳╳";
                
                vizFile << std::setw(15) << symbol;
            }
            vizFile << "\n";
        }

        vizFile << "\nLegend:\n";
        vizFile << "█  : Perfect correlation (1.0)\n";
        vizFile << "▓  : Strong positive correlation (0.7 to 1.0)\n";
        vizFile << "▒  : Moderate positive correlation (0.4 to 0.7)\n";
        vizFile << "░  : Weak positive correlation (0.2 to 0.4)\n";
        vizFile << "   : No correlation (-0.2 to 0.2)\n";
        vizFile << "╳  : Weak negative correlation (-0.4 to -0.2)\n";
        vizFile << "╳╳ : Moderate negative correlation (-0.7 to -0.4)\n";
        vizFile << "╳╳╳: Strong negative correlation (-1.0 to -0.7)\n";
        
        vizFile.close();
    }
};

class DataVisualizer {
private:
    std::vector<Person> data;
    int total_records = 0;

    std::vector<std::string> split(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            // Remove leading/trailing spaces
            token.erase(0, token.find_first_not_of(" "));
            token.erase(token.find_last_not_of(" ") + 1);
            tokens.push_back(token);
        }
        return tokens;
    }

    void createVisualizationDirectory() {
        #ifdef _WIN32
            system("mkdir visualizations 2> nul");
        #else
            system("mkdir -p visualizations");
        #endif
    }

public:
    bool loadDataFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }

        std::string line;
        // Skip header line if it exists
        std::getline(file, line);

        auto start = std::chrono::high_resolution_clock::now();
        int line_count = 0;

        while (std::getline(file, line)) {
            try {
                auto tokens = split(line, ',');
                if(tokens.size() >= 15) {
                    Person p;
                    p.age = std::stoi(tokens[0]);
                    p.workclass = tokens[1];
                    p.fnlwgt = std::stoi(tokens[2]);
                    p.education = tokens[3];
                    p.education_num = std::stoi(tokens[4]);
                    p.marital_status = tokens[5];
                    p.occupation = tokens[6];
                    p.relationship = tokens[7];
                    p.race = tokens[8];
                    p.gender = tokens[9];
                    p.capital_gain = std::stoi(tokens[10]);
                    p.capital_loss = std::stoi(tokens[11]);
                    p.hours_per_week = std::stoi(tokens[12]);
                    p.native_country = tokens[13];
                    p.income = tokens[14];
                    data.push_back(p);
                    line_count++;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing line " << line_count + 1 << ": " << e.what() << std::endl;
            }

            // Print progress every 1000 records
            if (line_count % 1000 == 0) {
                std::cout << "Processed " << line_count << " records...\r" << std::flush;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        total_records = line_count;
        std::cout << "\nProcessed " << line_count << " records in " 
                  << duration.count() / 1000.0 << " seconds" << std::endl;
        
        file.close();
        return true;
    }

    void generateAllVisualizations() {
        createVisualizationDirectory();
        
        std::cout << "Generating visualizations..." << std::endl;
        generateAgeDistribution();
        generateEducationDistribution();
        generateIncomeByEducation();
        generateGenderDistribution();
        generateWorkHoursDistribution();
        generateRaceDistribution();
        generateIncomeDistribution();
        generateSummaryStats();
        analyzeCorrelations();
    }

private:
    void analyzeCorrelations() {
        DataAnalyzer analyzer;
        analyzer.processData(data);
        analyzer.generateCorrelationMatrix("visualizations/correlation_matrix.csv");
        std::cout << "Generated correlation matrix in visualizations/correlation_matrix.csv" << std::endl;
        std::cout << "Generated ASCII visualization in visualizations/correlation_matrix.csv.txt" << std::endl;
    }
    void generateAgeDistribution() {
        std::map<int, int> age_groups;
        // Create age groups in 5-year intervals
        for(const auto& person : data) {
            int group = (person.age / 5) * 5;
            age_groups[group]++;
        }

        std::ofstream outFile("visualizations/age_distribution.csv");
        outFile << "Age_Group,Count,Percentage\n";
        for(const auto& pair : age_groups) {
            double percentage = (pair.second * 100.0) / total_records;
            outFile << pair.first << "-" << (pair.first + 4) << ","
                   << pair.second << "," 
                   << std::fixed << std::setprecision(2) << percentage << "\n";
        }
        outFile.close();
    }

    void generateEducationDistribution() {
        std::map<std::string, int> edu_dist;
        for(const auto& person : data) {
            edu_dist[person.education]++;
        }

        std::ofstream outFile("visualizations/education_distribution.csv");
        outFile << "Education,Count,Percentage\n";
        for(const auto& pair : edu_dist) {
            double percentage = (pair.second * 100.0) / total_records;
            outFile << pair.first << "," 
                   << pair.second << ","
                   << std::fixed << std::setprecision(2) << percentage << "\n";
        }
        outFile.close();
    }

    void generateIncomeByEducation() {
        std::map<std::string, std::pair<int, int>> income_by_edu;
        
        for(const auto& person : data) {
            if(person.income.find("<=50K") != std::string::npos) {
                income_by_edu[person.education].first++;
            } else {
                income_by_edu[person.education].second++;
            }
        }

        std::ofstream outFile("visualizations/income_by_education.csv");
        outFile << "Education,Income<=50K,Income>50K,Percentage>50K\n";
        for(const auto& pair : income_by_edu) {
            int total = pair.second.first + pair.second.second;
            double percentage = (pair.second.second * 100.0) / total;
            outFile << pair.first << "," 
                   << pair.second.first << "," 
                   << pair.second.second << ","
                   << std::fixed << std::setprecision(2) << percentage << "\n";
        }
        outFile.close();
    }

    void generateGenderDistribution() {
        std::map<std::string, std::pair<int, int>> gender_income;
        
        for(const auto& person : data) {
            if(person.income.find("<=50K") != std::string::npos) {
                gender_income[person.gender].first++;
            } else {
                gender_income[person.gender].second++;
            }
        }

        std::ofstream outFile("visualizations/gender_income_distribution.csv");
        outFile << "Gender,Income<=50K,Income>50K,Percentage>50K,Total\n";
        for(const auto& pair : gender_income) {
            int total = pair.second.first + pair.second.second;
            double percentage = (pair.second.second * 100.0) / total;
            outFile << pair.first << "," 
                   << pair.second.first << "," 
                   << pair.second.second << ","
                   << std::fixed << std::setprecision(2) << percentage << ","
                   << total << "\n";
        }
        outFile.close();
    }

    void generateWorkHoursDistribution() {
        std::map<int, int> hours_groups;
        // Group by 5-hour intervals
        for(const auto& person : data) {
            int group = (person.hours_per_week / 5) * 5;
            hours_groups[group]++;
        }

        std::ofstream outFile("visualizations/work_hours_distribution.csv");
        outFile << "Hours_Group,Count,Percentage\n";
        for(const auto& pair : hours_groups) {
            double percentage = (pair.second * 100.0) / total_records;
            outFile << pair.first << "-" << (pair.first + 4) << "," 
                   << pair.second << ","
                   << std::fixed << std::setprecision(2) << percentage << "\n";
        }
        outFile.close();
    }

    void generateRaceDistribution() {
        std::map<std::string, std::pair<int, int>> race_income;
        
        for(const auto& person : data) {
            if(person.income.find("<=50K") != std::string::npos) {
                race_income[person.race].first++;
            } else {
                race_income[person.race].second++;
            }
        }

        std::ofstream outFile("visualizations/race_income_distribution.csv");
        outFile << "Race,Income<=50K,Income>50K,Percentage>50K,Total\n";
        for(const auto& pair : race_income) {
            int total = pair.second.first + pair.second.second;
            double percentage = (pair.second.second * 100.0) / total;
            outFile << pair.first << "," 
                   << pair.second.first << "," 
                   << pair.second.second << ","
                   << std::fixed << std::setprecision(2) << percentage << ","
                   << total << "\n";
        }
        outFile.close();
    }

    void generateIncomeDistribution() {
        int income_below_50k = 0;
        int income_above_50k = 0;

        for(const auto& person : data) {
            if(person.income.find("<=50K") != std::string::npos) {
                income_below_50k++;
            } else {
                income_above_50k++;
            }
        }

        std::ofstream outFile("visualizations/income_distribution.csv");
        outFile << "Income_Category,Count,Percentage\n";
        
        double percent_below = (income_below_50k * 100.0) / total_records;
        double percent_above = (income_above_50k * 100.0) / total_records;
        
        outFile << "<=50K," << income_below_50k << "," 
                << std::fixed << std::setprecision(2) << percent_below << "\n";
        outFile << ">50K," << income_above_50k << "," 
                << std::fixed << std::setprecision(2) << percent_above << "\n";
        
        outFile.close();
    }

    void generateSummaryStats() {
        // Calculate basic statistics
        double avg_age = 0;
        double avg_hours = 0;
        int min_age = INT_MAX;
        int max_age = 0;
        int min_hours = INT_MAX;
        int max_hours = 0;

        for(const auto& person : data) {
            avg_age += person.age;
            avg_hours += person.hours_per_week;
            min_age = std::min(min_age, person.age);
            max_age = std::max(max_age, person.age);
            min_hours = std::min(min_hours, person.hours_per_week);
            max_hours = std::max(max_hours, person.hours_per_week);
        }

        avg_age /= total_records;
        avg_hours /= total_records;

        std::ofstream outFile("visualizations/summary_statistics.csv");
        outFile << "Metric,Value\n";
        outFile << "Total Records," << total_records << "\n";
        outFile << "Average Age," << std::fixed << std::setprecision(2) << avg_age << "\n";
        outFile << "Minimum Age," << min_age << "\n";
        outFile << "Maximum Age," << max_age << "\n";
        outFile << "Average Weekly Hours," << std::fixed << std::setprecision(2) << avg_hours << "\n";
        outFile << "Minimum Weekly Hours," << min_hours << "\n";
        outFile << "Maximum Weekly Hours," << max_hours << "\n";
        outFile.close();
    }
};

int main() {
    DataVisualizer visualizer;
    
    std::cout << "Reading data from adult.csv..." << std::endl;
    if (!visualizer.loadDataFromCSV("adult.csv")) {
        return 1;
    }

    visualizer.generateAllVisualizations();
    std::cout << "\nVisualization files have been generated in the 'visualizations' directory." << std::endl;
    return 0;
}
