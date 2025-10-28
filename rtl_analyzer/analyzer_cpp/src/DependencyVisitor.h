#pragma once
#include "DataModel.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/statements/ConditionalStatements.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"

using AnalysisResultMap = std::map<std::string, VariableInfo>;

// 前置声明辅助函数
bool isEnumConstant(const slang::ast::Symbol& symbol);

class DependencyVisitor : public slang::ast::ASTVisitor<DependencyVisitor, true, true> {
public:
    DependencyVisitor();

    // 简化模板处理
    template<typename T>
    void handle(const T& node) {
        visitDefault(node);
    }
    
    // 显式声明需要处理的节点类型
    void handle(const slang::ast::VariableSymbol& symbol);
    void handle(const slang::ast::PortSymbol& symbol);
    void handle(const slang::ast::AssignmentExpression& expr);
    void handle(const slang::ast::ConditionalStatement& stmt);
    void handle(const slang::ast::CaseStatement& stmt);
    void handle(const slang::ast::InstanceSymbol& symbol);
    void handle(const slang::ast::ProceduralBlockSymbol& symbol);
    
    void postProcess();

    const AnalysisResultMap& getResults() const { return results; }

private:
    VariableInfo& getOrAddVariable(const slang::ast::Symbol& symbol);
    
    // 新增方法声明
    void handleMemberAssignment(const slang::ast::AssignmentExpression& expr, 
                               const slang::ast::MemberAccessExpression& memberExpr);
    VariableInfo& getOrAddVariableByName(const std::string& fullName);
    std::string getFullMemberPath(const slang::ast::Expression& expr);
    
    std::string extractCaseItemExpression(const slang::ast::Expression& caseExpr, const slang::ast::CaseStatement::ItemGroup& item);
    std::vector<ConditionPath> buildCasePaths(const slang::ast::CaseStatement& stmt, const ConditionPath& parentPath);
    bool isControlVariable(const std::string& varName);
    
    const slang::ast::ProceduralBlockSymbol* currentProcBlock = nullptr;  
    std::vector<ConditionPath> pathStack;
    AnalysisResultMap results;
};