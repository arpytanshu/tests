#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>

using namespace std;


class Value;
struct row{float x[2]; int y;};
struct children{ Value* left; Value* right;};

vector<row> parse_csv(string filename){
    ifstream file(filename);
    vector<row> data;
    string line;
    while (getline(file, line)){
        row r;
        stringstream ss(line);
        string token;
        int i = 0;
        while (getline(ss, token, ',')){
            if (i < 2){
                r.x[i] = stof(token);
            }
            else{
                r.y = stoi(token);
            }
            i++;
        }
        data.push_back(r);
    }
    return data;
}

string DATA_PATH = "data.csv";


class Value{
    float data;
    float grad;
    string op;
    children child;
    public:
        Value(float data, string op="input", children child={NULL, NULL}){
            this->data = data;
            this->op = op;
            this->child = child;
        }
        void backward(float grad=1.0){
            this->grad += grad;
            if (this->op == "add"){
                this->child.left->backward(grad);
                this->child.right->backward(grad);
            }
            else if (this->op == "mul"){
                this->child.left->backward(grad * this->child.right->data);
                this->child.right->backward(grad * this->child.left->data);
            }
        }
        void zero_grad(){
            this->grad = 0.0;
        }
        Value* add(Value* other){
            Value* out = new Value(this->data + other->data, "add", {this, other});
            return out;
        }
        Value* mul(Value* other){
            Value* out = new Value(this->data * other->data, "mul", {this, other});
            return out;
        }
        Value* relu(){
            Value* out = new Value(max(0.0, this->data), "relu", {this, NULL});
            return out;
        }
        void __repr__(){
            cout << "Value(" << this->data << ", grad=" << this->grad << ")" << endl;
        }

        std::ostream& operator<< (std::ostream &out, AutoData const& data) {
            out << data.getmpg() << ':';
            out << data.getcylinders() << ':';
            // and so on... 
            return out;
}
};


int main(){
    vector<row> data = parse_csv(DATA_PATH);
    // for(int i=0; i< data.size(); i++){
    //     cout << data[i].x[0] << " " << data[i].x[1] << " " << data[i].y << endl;
    // }

    return 0;
}