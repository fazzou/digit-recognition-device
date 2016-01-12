#include <iostream>
#include <stack>
#include <set>
#include <map>
#include <assert.h>
#include <vector>
#include <sstream>
#include <cstdlib>

using namespace std;

bool isOperator(string toCheck, map<string,int> * operators){
    return operators->count(toCheck);
}

int getOperatorPriority(string op, map<string,int> *operators){
    return operators->at(op);
}

int priorityComparator(string firstOp, string secondOp, map<string, int> *operators){
    return operators->at(firstOp) - operators->at(secondOp);
}

string intToString (int a)
{
    ostringstream temp;
    temp<<a;
    return temp.str();
}

int evaluateRPNFormula(vector<string> toEvaluate, map<string, int> *operators){
    vector<string>::iterator it;
    int returnValue = 0;
    stack<string> st;
    int a,b;

    for(it=toEvaluate.begin(); it<toEvaluate.end();it++){
        string token = *it;
        if(!isOperator(token, operators)){
            st.push(token);
        }
        else{
            if(st.empty()){
                return 0;
            }
            else{
                a = atoi(st.top().c_str());
                st.pop();
            }

            if(st.empty()){
                return 0;
            }
            else{
                b = atoi(st.top().c_str());
                st.pop();
            }

            int value = 0;

            if(token == "+"){
                value = a+b;
            }
            else if(token == "-"){
                value = b - a;
            }
            else if(token == "*"){
                value = a * b;
            }

            st.push(intToString(value));
        }
    }

    if(st.empty()){
        return 0;
    }

    returnValue = atoi(st.top().c_str());
    return returnValue;
}

vector<string> convertToRPN(vector<string> toConvert, map<string,int> *operatorsMap){
    vector<string> convertedStrings;
    stack<string> stackOp;
    vector<string>::iterator it;

    for(it=toConvert.begin(); it<toConvert.end();it++){
        string token = *it;
        if(isOperator(token, operatorsMap)){
            while(!stackOp.empty() && isOperator(stackOp.top(), operatorsMap)){
                if(priorityComparator(token, stackOp.top(), operatorsMap)<=0){
                    convertedStrings.push_back(stackOp.top());
                    stackOp.pop();
                    continue;
                }
                break;
            }
            stackOp.push(token);
        }
        else if(token=="("){
            stackOp.push(token);
        }
        else if(token==")"){
            while(!stackOp.empty() && !(stackOp.top() == "(")){
                convertedStrings.push_back(stackOp.top());
                stackOp.pop();
            }
            if(stackOp.empty()){
                return vector<string>();
            } else{
                stackOp.pop();
            }
        }
        else{
            convertedStrings.push_back(token);
        }
    }

    while(!stackOp.empty()){
        convertedStrings.push_back(stackOp.top());
        stackOp.pop();
    }

    return convertedStrings;
}

int convertAndEvaluate(vector<string> formula, map<string, int> *operators){
    vector<string> convertedToRPN = convertToRPN(formula, operators);
    cout<<"konwersja zakonczona sukcesem"<<endl;
    int result = evaluateRPNFormula(convertedToRPN, operators);
    cout<<"ewaluacja zakonczona sukcesem"<<endl;
    return result;
}

vector<string> convertStringIntoStringVector(string s){
    vector<string> internal;
    stringstream ss(s);
    string tok;

    while(getline(ss, tok, ' ')) {
      internal.push_back(tok);
    }

    vector<string>::iterator it;

    return internal;
}

void tests(map<string,int> * operators){
    assert(isOperator("+", operators)==true);
    assert(isOperator("-", operators)==true);
    assert(isOperator("*", operators)==true);
    assert(isOperator("f", operators)==false);
    assert(isOperator("]", operators)==false);

    assert(getOperatorPriority("+", operators)==1);
    assert(getOperatorPriority("-", operators)==1);
    assert(getOperatorPriority("*", operators)==5);

    assert(intToString(20) == "20");
    assert(intToString(744) == "744");
    assert(intToString(22) == "22");
    assert(intToString(-2) == "-2");

    vector<string> formula;
    formula.push_back("10");
    formula.push_back("3");
    formula.push_back("*");
    formula.push_back("10");
    formula.push_back("-");

    assert(evaluateRPNFormula(formula, operators)== 20);

    vector<string> formula2;
    formula2.push_back("5");
    formula2.push_back("3");
    formula2.push_back("*");
    formula2.push_back("16");
    formula2.push_back("-");

    assert(evaluateRPNFormula(formula2, operators)== -1);

    map<string, int> operators;
    operators["-"] = 1;
    operators["+"] = 1;
    operators["*"] = 5;

    assert(convertAndEvaluate(convertStringIntoStringVector("( ( 3 + 2 ) )"), &operators) == 5);
    assert(convertAndEvaluate(convertStringIntoStringVector("( 3 + 2 ) - 8 + 2"), &operators) == -1);
    assert(convertAndEvaluate(convertStringIntoStringVector(" 3 + 2 + 12 * 2"), &operators) == 29);
    assert(convertAndEvaluate(convertStringIntoStringVector(" 10 * 5 - 3 * 10"), &operators) == 20);
    assert(convertAndEvaluate(convertStringIntoStringVector(" 10 * ( 10 - 3 )"), &operators) == 70);
    assert(convertAndEvaluate(convertStringIntoStringVector(" ( ( 10 - 9 ) + ( 3 - 2 ) * ( 10 - 3 ) )"), &operators) == 8);

    //negativ cases
    assert(convertAndEvaluate(convertStringIntoStringVector(" 23 - 33 - ) * ) + ) "), &operators)==0);
    assert(convertAndEvaluate(convertStringIntoStringVector(" ) 0 * + 10 (8  "), &operators)==0);
    assert(convertAndEvaluate(convertStringIntoStringVector(" 10 + + 3 "), &operators) == 13);
}
