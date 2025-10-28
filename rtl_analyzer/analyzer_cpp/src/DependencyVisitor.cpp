#include "DependencyVisitor.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/expressions/OperatorExpressions.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/expressions/SelectExpressions.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/Type.h"
#include "slang/ast/types/AllTypes.h"
#include "slang/text/SourceManager.h"
#include <sstream>
#include <iostream>

// ============================================================================
// 辅助函数
// ============================================================================

// 检查是否为编译时常量
bool isCompileTimeConstant(const slang::ast::Symbol& symbol) {
    return symbol.kind == slang::ast::SymbolKind::EnumValue ||
           symbol.kind == slang::ast::SymbolKind::Parameter; // localparam 也是 Parameter
}

// 自增操作检测
bool isIncrementOperation(const slang::ast::Expression& expr, const slang::ast::Symbol& lhsSymbol) {
    if (const auto* binaryExpr = expr.as_if<slang::ast::BinaryExpression>()) {
        if (binaryExpr->op == slang::ast::BinaryOperator::Add) {
            // 检查左操作数是否是同一个信号
            if (const auto* leftValue = binaryExpr->left().as_if<slang::ast::NamedValueExpression>()) {
                if (&leftValue->symbol == &lhsSymbol) {
                    // 检查右操作数是否是常量
                    if (binaryExpr->right().as_if<slang::ast::IntegerLiteral>()) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// 枚举常量检测
bool isEnumConstant(const slang::ast::Symbol& symbol) {
    return symbol.kind == slang::ast::SymbolKind::EnumValue;
}

// 端口方向转换
static std::string directionToString(slang::ast::ArgumentDirection dir) {
    switch (dir) {
        case slang::ast::ArgumentDirection::In: return "input";
        case slang::ast::ArgumentDirection::Out: return "output";
        case slang::ast::ArgumentDirection::InOut: return "inout";
        default: return "unknown";
    }
}

// ============================================================================
// DataSignalVisitor - 提取数据信号，过滤枚举常量
// ============================================================================

class DataSignalVisitor : public slang::ast::ASTVisitor<DataSignalVisitor, true, true> {
public:
    template<typename T> void handle(const T& node) { visitDefault(node); }

    void handle(const slang::ast::NamedValueExpression& expr) {
        const slang::ast::Symbol* symbol = &expr.symbol;
        if (symbol && !symbol->isType()) {
            if (!isCompileTimeConstant(*symbol)) {
                std::string signalName = symbol->getHierarchicalPath();
                signals.insert(signalName);
            }
        }
    }

    void handle(const slang::ast::MemberAccessExpression& expr) {
        std::string fullPath = getFullMemberPath(expr);
        if (!fullPath.empty()) {
            std::cout << "[DEBUG] DataSignalVisitor found member: " << fullPath << std::endl;
            signals.insert(fullPath);
            
            // // 同时记录父级
            // std::string parentPath = getFullMemberPath(expr.value());
            // if (!parentPath.empty()) {
            //     std::cout << "[DEBUG] DataSignalVisitor found parent: " << parentPath << std::endl;
            //     signals.insert(parentPath);
            // }
        }
    }

    std::string getFullMemberPath(const slang::ast::Expression& expr) {
        if (const auto* namedExpr = expr.as_if<slang::ast::NamedValueExpression>()) {
            return std::string(namedExpr->symbol.getHierarchicalPath());
        }
        if (const auto* memberExpr = expr.as_if<slang::ast::MemberAccessExpression>()) {
            std::string base = getFullMemberPath(memberExpr->value());
            if (!base.empty()) {
                return base + "." + std::string(memberExpr->member.name);
            }
        }
        if (const auto* selectExpr = expr.as_if<slang::ast::ElementSelectExpression>()) {
            std::string base = getFullMemberPath(selectExpr->value());
            if (!base.empty()) {
                return base + "[*]";
            }
        }
        return "";
    }

    std::set<std::string> signals;
};

// ============================================================================
// ConditionClauseVisitor - 分析条件子句，过滤枚举常量
// ============================================================================

class ConditionClauseVisitor : public slang::ast::ASTVisitor<ConditionClauseVisitor, true, true> {
public:
    template<typename T> void handle(const T& node) { 
        visitDefault(node); 
    }

    void handle(const slang::ast::NamedValueExpression& expr) {
        const slang::ast::Symbol* symbol = &expr.symbol;
        if (!symbol || symbol->isType()) return;

        std::string fullName = symbol->getHierarchicalPath();
        bool isParamOrEnum = isCompileTimeConstant(*symbol);

        if (!currentExpression.empty()) {
            currentExpression += " ";
        }
        currentExpression += fullName;

        if (isParamOrEnum) {
            involvedParameters.insert(fullName);
        } else {
            involvedSignals.insert(fullName);
        }
    }

    void handle(const slang::ast::MemberAccessExpression& expr) {
        std::string fullPath = getFullMemberPath(expr);
        if (!fullPath.empty()) {
            bool isParamOrEnum = false;
            if (const slang::ast::Symbol* memberSym = expr.getSymbolReference()) {
                isParamOrEnum = isCompileTimeConstant(*memberSym);
            }
            
            if (!currentExpression.empty()) {
                currentExpression += " ";
            }
            currentExpression += fullPath;
            
            if (isParamOrEnum) {
                involvedParameters.insert(fullPath);
            } else {
                involvedSignals.insert(fullPath);
            }
            
            // 同时记录父级结构体/联合体的依赖
            std::string parentPath = getFullMemberPath(expr.value());
            if (!parentPath.empty()) {
                bool parentIsParamOrEnum = false;
                if (const auto* namedExpr = expr.value().as_if<slang::ast::NamedValueExpression>()) {
                    parentIsParamOrEnum = isCompileTimeConstant(namedExpr->symbol);
                }
                
                if (parentIsParamOrEnum) {
                    involvedParameters.insert(parentPath);
                } else {
                    involvedSignals.insert(parentPath);
                }
            }
        } else {
            visitDefault(expr);
        }
    }

    void handle(const slang::ast::BinaryExpression& expr) {
        expr.left().visit(*this);
        
        std::string opStr;
        switch (expr.op) {
            case slang::ast::BinaryOperator::Equality: opStr = " == "; break;
            case slang::ast::BinaryOperator::Inequality: opStr = " != "; break;
            case slang::ast::BinaryOperator::LogicalAnd: opStr = " && "; break;
            case slang::ast::BinaryOperator::LogicalOr: opStr = " || "; break;
            case slang::ast::BinaryOperator::GreaterThanEqual: opStr = " >= "; break;
            case slang::ast::BinaryOperator::GreaterThan: opStr = " > "; break;
            case slang::ast::BinaryOperator::LessThanEqual: opStr = " <= "; break;
            case slang::ast::BinaryOperator::LessThan: opStr = " < "; break;
            default: opStr = " op "; break;
        }
        currentExpression += opStr;
        
        expr.right().visit(*this);
    }

    void handle(const slang::ast::UnaryExpression& expr) {
        std::string opStr;
        switch (expr.op) {
            case slang::ast::UnaryOperator::LogicalNot: opStr = "!"; break;
            case slang::ast::UnaryOperator::BitwiseAnd: opStr = "&"; break;
            case slang::ast::UnaryOperator::BitwiseOr: opStr = "|"; break;
            case slang::ast::UnaryOperator::BitwiseXor: opStr = "^"; break;
            default: opStr = "unary_op "; break;
        }
        currentExpression += opStr;
        
        expr.operand().visit(*this);
    }

    void handle(const slang::ast::IntegerLiteral& literal) {
        if (!currentExpression.empty()) {
            currentExpression += " ";
        }
        currentExpression += literal.getValue().toString();
    }

    void handle(const slang::ast::ElementSelectExpression& expr) {
        expr.value().visit(*this);
        currentExpression += "[";
        expr.selector().visit(*this);
        currentExpression += "]";
    }

    void handle(const slang::ast::RangeSelectExpression& expr) {
        expr.value().visit(*this);
        currentExpression += "[";
        expr.left().visit(*this);
        currentExpression += ":";
        expr.right().visit(*this);
        currentExpression += "]";
    }

    std::string getExpressionString() const {
        return currentExpression;
    }

    std::set<std::string> getInvolvedSignals() const {
        return involvedSignals;
    }

    std::set<std::string> getInvolvedParameters() const {
        return involvedParameters;
    }

    void reset() {
        currentExpression.clear();
        involvedSignals.clear();
        involvedParameters.clear();
    }

    std::string getFullMemberPath(const slang::ast::Expression& expr) {
        if (const auto* namedExpr = expr.as_if<slang::ast::NamedValueExpression>()) {
            return std::string(namedExpr->symbol.getHierarchicalPath());
        }
        if (const auto* memberExpr = expr.as_if<slang::ast::MemberAccessExpression>()) {
            std::string base = getFullMemberPath(memberExpr->value());
            if (!base.empty()) {
                return base + "." + std::string(memberExpr->member.name);
            }
        }
        if (const auto* selectExpr = expr.as_if<slang::ast::ElementSelectExpression>()) {
            std::string base = getFullMemberPath(selectExpr->value());
            if (!base.empty()) {
                return base + "[*]";
            }
        }
        return "";
    }

private:
    std::string currentExpression;
    std::set<std::string> involvedSignals;
    std::set<std::string> involvedParameters;
};

// ============================================================================
// CaseItemExpressionVisitor - 处理 case 项表达式
// ============================================================================

class CaseItemExpressionVisitor : public slang::ast::ASTVisitor<CaseItemExpressionVisitor, true, true> {
public:
    template<typename T> void handle(const T& node) { 
        visitDefault(node); 
    }

    void handle(const slang::ast::NamedValueExpression& expr) {
        const slang::ast::Symbol* symbol = &expr.symbol;
        if (symbol && !symbol->isType()) {
            std::string signalName = symbol->getHierarchicalPath();
            if (!currentExpression.empty()) {
                currentExpression += " ";
            }
            currentExpression += signalName;
        }
    }

    void handle(const slang::ast::IntegerLiteral& literal) {
        if (!currentExpression.empty()) {
            currentExpression += " ";
        }
        currentExpression += literal.getValue().toString();
    }

    void handle(const slang::ast::BinaryExpression& expr) {
        if (expr.op == slang::ast::BinaryOperator::LogicalOr) {
            expr.left().visit(*this);
            currentExpression += " || ";
            expr.right().visit(*this);
        } else {
            visitDefault(expr);
        }
    }

    void handle(const slang::ast::UnaryExpression& expr) {
        if (expr.op == slang::ast::UnaryOperator::LogicalNot) {
            currentExpression += "!";
            expr.operand().visit(*this);
        } else {
            visitDefault(expr);
        }
    }

    std::string getExpressionString() const {
        return currentExpression;
    }

    void reset() {
        currentExpression.clear();
    }

private:
    std::string currentExpression;
};

// ============================================================================
// DependencyVisitor 主实现
// ============================================================================

DependencyVisitor::DependencyVisitor() {
    pathStack.push_back({});
}

// 获取或创建变量信息
VariableInfo& DependencyVisitor::getOrAddVariable(const slang::ast::Symbol& symbol) {  
    std::string path = symbol.getHierarchicalPath();  
    if (results.find(path) == results.end()) {  
        VariableInfo info;  
        info.fullName = path;  

        if (const auto* portSymbol = symbol.as_if<slang::ast::PortSymbol>()) {  
            info.direction = directionToString(portSymbol->direction);  
            if (portSymbol->internalSymbol) {  
                const auto* internalValue = portSymbol->internalSymbol->as_if<slang::ast::ValueSymbol>();  
                if (internalValue) {  
                    const slang::ast::Type& type = internalValue->getType();  
                    info.type = type.toString();  
                    info.bitWidth = type.getBitWidth();  
                }  
            }  
        } else if (const auto* valueSymbol = symbol.as_if<slang::ast::ValueSymbol>()) {  
            const slang::ast::Type& type = valueSymbol->getType();  
            info.type = type.toString();  
            info.bitWidth = type.getBitWidth();  
        }  

        const slang::ast::Scope* scope = symbol.getParentScope();  
        if (scope) {  
            auto& comp = scope->getCompilation();  
            auto* sm = comp.getSourceManager();  
            if (sm && symbol.location) {  
                info.fileName = std::string(sm->getFileName(symbol.location));  
                info.line = sm->getLineNumber(symbol.location);  
            }  
        }  
        results[path] = info;  
    }  
    return results.at(path);  
}

// 通过名称获取或创建变量信息
VariableInfo& DependencyVisitor::getOrAddVariableByName(const std::string& fullName) {
    if (results.find(fullName) == results.end()) {
        VariableInfo info;
        info.fullName = fullName;
        results[fullName] = info;
    }
    return results.at(fullName);
}

// 获取完整的成员访问路径
std::string DependencyVisitor::getFullMemberPath(const slang::ast::Expression& expr) {
    if (const auto* namedExpr = expr.as_if<slang::ast::NamedValueExpression>()) {
        return std::string(namedExpr->symbol.getHierarchicalPath());
    }
    if (const auto* memberExpr = expr.as_if<slang::ast::MemberAccessExpression>()) {
        std::string base = getFullMemberPath(memberExpr->value());
        if (!base.empty()) {
            // 获取成员名称
            std::string memberName = std::string(memberExpr->member.name);
            return base + "." + memberName;
        }
    }
    return "";
}

// 处理成员访问赋值
// 处理成员访问赋值
void DependencyVisitor::handleMemberAssignment(const slang::ast::AssignmentExpression& expr, 
                                              const slang::ast::MemberAccessExpression& memberExpr) {
    std::cout << "[DEBUG] handleMemberAssignment called!" << std::endl;
    
    // 获取完整的成员路径
    std::string fullMemberPath = getFullMemberPath(memberExpr);
    std::cout << "[DEBUG] Full member path: " << fullMemberPath << std::endl;
    
    if (fullMemberPath.empty()) {
        std::cout << "[DEBUG] Empty member path, skipping" << std::endl;
        visitDefault(expr);
        return;
    }
    
    // 为成员创建变量信息
    VariableInfo& memberInfo = getOrAddVariableByName(fullMemberPath);
    std::cout << "[DEBUG] Created variable info for: " << fullMemberPath << std::endl;
    
    // 分析右侧表达式
    DataSignalVisitor rhsVisitor;
    expr.right().visit(rhsVisitor);
    
    std::cout << "[DEBUG] RHS signals: ";
    for (const auto& sig : rhsVisitor.signals) {
        std::cout << sig << " ";
    }
    std::cout << std::endl;
    
    // 创建赋值记录
    AssignmentInfo assignInfo;
    assignInfo.path = pathStack.back();
    assignInfo.drivingSignals = rhsVisitor.signals;
    assignInfo.type = "member_assignment";
    assignInfo.logicType = "combinational";
    assignInfo.conditionDepth = pathStack.size() - 1;
    
    memberInfo.assignments.insert(assignInfo);
    std::cout << "[DEBUG] Added assignment to member: " << fullMemberPath << std::endl;
    
    // // 同时为父级结构体/联合体创建赋值记录
    // std::string parentPath = getFullMemberPath(memberExpr.value());
    // if (!parentPath.empty()) {
    //     std::cout << "[DEBUG] Parent path: " << parentPath << std::endl;
        
    //     VariableInfo& parentInfo = getOrAddVariableByName(parentPath);
        
    //     AssignmentInfo parentAssignInfo;
    //     parentAssignInfo.path = pathStack.back();
    //     parentAssignInfo.drivingSignals = {fullMemberPath};
    //     parentAssignInfo.type = "structural_parent";
    //     parentAssignInfo.logicType = "combinational";
    //     parentAssignInfo.conditionDepth = pathStack.size() - 1;
        
    //     parentInfo.assignments.insert(parentAssignInfo);
    //     std::cout << "[DEBUG] Added parent assignment to: " << parentPath << std::endl;
    // }
    
    visitDefault(expr);
}

// 提取 case 项表达式
std::string DependencyVisitor::extractCaseItemExpression(const slang::ast::Expression& caseExpr, 
                                                        const slang::ast::CaseStatement::ItemGroup& item) {
    CaseItemExpressionVisitor visitor;
    
    if (item.expressions.empty()) {
        return "default";
    }
    
    // 获取case表达式的字符串表示
    ConditionClauseVisitor caseExprVisitor;
    caseExpr.visit(caseExprVisitor);
    std::string caseExprStr = caseExprVisitor.getExpressionString();
    
    // 处理多个表达式（OR关系）
    std::string result;
    for (size_t i = 0; i < item.expressions.size(); ++i) {
        if (i > 0) {
            result += " || ";
        }
        visitor.reset();
        item.expressions[i]->visit(visitor);
        result += caseExprStr + " == " + visitor.getExpressionString();
    }
    
    return result;
}

// 构建 case 路径
std::vector<ConditionPath> DependencyVisitor::buildCasePaths(const slang::ast::CaseStatement& stmt, 
                                                            const ConditionPath& parentPath) {
    std::vector<ConditionPath> casePaths;
    
    ConditionClauseVisitor caseExprVisitor;
    stmt.expr.visit(caseExprVisitor);
    
    ConditionExpression caseExpr;
    caseExpr.expression = caseExprVisitor.getExpressionString();
    caseExpr.involvedSignals = caseExprVisitor.getInvolvedSignals();
    caseExpr.involvedParameters = caseExprVisitor.getInvolvedParameters();
    bool hasDefault = false;
    
    // 处理每个case项
    for (const auto& item : stmt.items) {
        std::string itemExpression = extractCaseItemExpression(stmt.expr, item);
        
        if (itemExpression == "default") {
            hasDefault = true;
            continue;
        }
        
        ConditionExpression fullCondition;
        fullCondition.expression = itemExpression;
        fullCondition.involvedSignals = caseExpr.involvedSignals;
        
        // 添加item表达式中涉及的信号
        for (const auto& expr : item.expressions) {
            ConditionClauseVisitor itemVisitor;
            expr->visit(itemVisitor);
            auto signals = itemVisitor.getInvolvedSignals();
            fullCondition.involvedSignals.insert(signals.begin(), signals.end());
        }
        
        ConditionClause conditionClause;
        conditionClause.expr = fullCondition;
        conditionClause.polarity = true;
        
        ConditionPath itemPath = parentPath;
        itemPath.insert(conditionClause);
        casePaths.push_back(itemPath);
    }
    
    // 处理default情况
    if (hasDefault || stmt.defaultCase) {
        ConditionPath defaultPath = parentPath;
        
        for (const auto& item : stmt.items) {
            std::string itemExpression = extractCaseItemExpression(stmt.expr, item);
            if (itemExpression == "default") continue;
            
            ConditionExpression fullCondition;
            fullCondition.expression = itemExpression;
            fullCondition.involvedSignals = caseExpr.involvedSignals;
            
            for (const auto& expr : item.expressions) {
                ConditionClauseVisitor itemVisitor;
                expr->visit(itemVisitor);
                auto signals = itemVisitor.getInvolvedSignals();
                fullCondition.involvedSignals.insert(signals.begin(), signals.end());
            }
            
            ConditionClause conditionClause;
            conditionClause.expr = fullCondition;
            conditionClause.polarity = false;
            
            defaultPath.insert(conditionClause);
        }
        
        casePaths.push_back(defaultPath);
    }
    
    return casePaths;
}

// ============================================================================
// 主要处理函数
// ============================================================================

void DependencyVisitor::handle(const slang::ast::VariableSymbol& symbol) {
    getOrAddVariable(symbol);
    visitDefault(symbol);
}

void DependencyVisitor::handle(const slang::ast::PortSymbol& symbol) {
    getOrAddVariable(symbol);
    visitDefault(symbol);
}

void DependencyVisitor::handle(const slang::ast::AssignmentExpression& expr) {
    std::cout << "[DEBUG] Processing assignment expression" << std::endl;
    
    // 首先检查是否为成员访问表达式
    if (const auto* memberExpr = expr.left().as_if<slang::ast::MemberAccessExpression>()) {
        std::cout << "[DEBUG] Found member access on LHS, handling member assignment" << std::endl;
        handleMemberAssignment(expr, *memberExpr);
        return;
    }
    
    // 如果不是成员访问，再检查简单符号
    const slang::ast::Symbol* lhsSymbol = expr.left().getSymbolReference();
    if (!lhsSymbol) {
        visitDefault(expr);
        return;
    }
    
    // 原有的简单符号处理逻辑
    std::string lhsName = lhsSymbol->getHierarchicalPath();
    std::cout << "[DEBUG] Simple assignment to: " << lhsName << std::endl;
    
    VariableInfo& lhsInfo = getOrAddVariable(*lhsSymbol);
        
    bool isIncrement = isIncrementOperation(expr.right(), *lhsSymbol);

    DataSignalVisitor rhsVisitor;
    if (!isIncrement) {
        expr.right().visit(rhsVisitor);
    }

    AssignmentInfo assignInfo;
    assignInfo.path = pathStack.back();
    assignInfo.drivingSignals = rhsVisitor.signals;
    if (isIncrement) {
        assignInfo.type = "increment";
    } else if (rhsVisitor.signals.empty()) {
        assignInfo.type = "constant";
    } else {
        assignInfo.type = "direct";
    }
    
    // 时序逻辑检测
    if (currentProcBlock) {  
        if (currentProcBlock->procedureKind == slang::ast::ProceduralBlockKind::AlwaysFF) {  
            assignInfo.logicType = "sequential";  
        }   
        else if (currentProcBlock->procedureKind == slang::ast::ProceduralBlockKind::AlwaysComb ||  
                 currentProcBlock->procedureKind == slang::ast::ProceduralBlockKind::AlwaysLatch) {  
            assignInfo.logicType = "combinational";  
        }  
        else if (currentProcBlock->procedureKind == slang::ast::ProceduralBlockKind::Always) {  
            if (expr.isNonBlocking()) {  
                assignInfo.logicType = "sequential";  
            } else {  
                assignInfo.logicType = "combinational";  
            }  
        }  
        else if (currentProcBlock->procedureKind == slang::ast::ProceduralBlockKind::Initial ||  
                 currentProcBlock->procedureKind == slang::ast::ProceduralBlockKind::Final) {  
            assignInfo.logicType = "initialization";  
        }  
        else {  
            assignInfo.logicType = "combinational";  
        }  
    } else {  
        assignInfo.logicType = "combinational";  
    }  
    
    assignInfo.conditionDepth = pathStack.size() - 1;

    const slang::ast::Scope* scope = lhsSymbol->getParentScope();
    if (scope) {
        auto& comp = scope->getCompilation();
        auto* sm = comp.getSourceManager();
        if (sm && expr.sourceRange.start()) {
            assignInfo.file = std::string(sm->getFileName(expr.sourceRange.start()));
            assignInfo.line = sm->getLineNumber(expr.sourceRange.start());
        }
    }
    lhsInfo.assignments.insert(assignInfo);
    
    visitDefault(expr);
}

void DependencyVisitor::handle(const slang::ast::ConditionalStatement& stmt) {
    ConditionPath parentPath = pathStack.back();
    
    for (size_t i = 0; i < stmt.conditions.size(); ++i) {
        const auto& cond = stmt.conditions[i];
        
        ConditionClauseVisitor clauseVisitor;
        cond.expr->visit(clauseVisitor);

        ConditionExpression condExpr;
        condExpr.expression = clauseVisitor.getExpressionString();
        condExpr.involvedSignals = clauseVisitor.getInvolvedSignals();
        condExpr.involvedParameters = clauseVisitor.getInvolvedParameters();
        ConditionClause trueClause;
        trueClause.expr = condExpr;
        trueClause.polarity = true;

        ConditionPath truePath = parentPath;
        truePath.insert(trueClause);
        pathStack.push_back(truePath);
        
        stmt.ifTrue.visit(*this);
        pathStack.pop_back();

        ConditionClause falseClause;
        falseClause.expr = condExpr;
        falseClause.polarity = false;
        parentPath.insert(falseClause);
    }

    if (stmt.ifFalse) {
        pathStack.push_back(parentPath);
        stmt.ifFalse->visit(*this);
        pathStack.pop_back();
    }
}

void DependencyVisitor::handle(const slang::ast::CaseStatement& stmt) {
    ConditionPath parentPath = pathStack.back();
    
    std::vector<ConditionPath> casePaths = buildCasePaths(stmt, parentPath);
    
    size_t pathIndex = 0;
    for (const auto& item : stmt.items) {
        if (pathIndex < casePaths.size()) {
            pathStack.push_back(casePaths[pathIndex]);
            if (item.stmt) {
                item.stmt->visit(*this);
            }
            pathStack.pop_back();
            pathIndex++;
        }
    }
    
    if (stmt.defaultCase && pathIndex < casePaths.size()) {
        pathStack.push_back(casePaths[pathIndex]);
        stmt.defaultCase->visit(*this);
        pathStack.pop_back();
    }
}

void DependencyVisitor::handle(const slang::ast::InstanceSymbol& symbol) {
    std::string instanceFile;
    int instanceLine = 0;
    
    const slang::ast::Scope* scope = symbol.getParentScope();
    if (scope) {
        auto& comp = scope->getCompilation();
        auto* sm = comp.getSourceManager();
        if (sm && symbol.location) {
            instanceFile = std::string(sm->getFileName(symbol.location));
            instanceLine = sm->getLineNumber(symbol.location);
        }
    }

    for (auto* portConnection : symbol.getPortConnections()) {
        const slang::ast::Symbol& internalPort = portConnection->port;
        const slang::ast::Expression* externalExpr = portConnection->getExpression();
        
        VariableInfo& internalPortInfo = getOrAddVariable(internalPort);
        
        if (!externalExpr) continue;
        
        DataSignalVisitor externalSignalVisitor;
        externalExpr->visit(externalSignalVisitor);

        for (const auto& externalSignalName : externalSignalVisitor.signals) {
            if (results.count(externalSignalName)) {
                VariableInfo& externalInfo = results[externalSignalName];
                
                auto direction = internalPort.as<slang::ast::PortSymbol>().direction;
                
                if (direction != slang::ast::ArgumentDirection::In) { 
                    AssignmentInfo assignInfo;
                    assignInfo.path = pathStack.back();
                    assignInfo.drivingSignals = {internalPortInfo.fullName};
                    assignInfo.file = instanceFile;
                    assignInfo.line = instanceLine;
                    assignInfo.type = "port_connection";
                    assignInfo.logicType = "combinational";
                    assignInfo.conditionDepth = pathStack.size() - 1;
                    externalInfo.assignments.insert(assignInfo);
                    
                    internalPortInfo.fanOut.insert(externalInfo.fullName);
                }
                if (direction != slang::ast::ArgumentDirection::Out) { 
                    AssignmentInfo assignInfo;
                    assignInfo.path = pathStack.back();
                    assignInfo.drivingSignals = std::set<std::string>{externalSignalName};
                    assignInfo.file = instanceFile;
                    assignInfo.line = instanceLine;
                    assignInfo.type = "port_connection";
                    assignInfo.logicType = "combinational";
                    assignInfo.conditionDepth = pathStack.size() - 1;
                    internalPortInfo.assignments.insert(assignInfo);
                    
                    externalInfo.fanOut.insert(internalPortInfo.fullName);
                }
            }
        }
    }

    visitDefault(symbol);
}

void DependencyVisitor::handle(const slang::ast::ProceduralBlockSymbol& symbol) {  
    auto* prevBlock = currentProcBlock;  
    currentProcBlock = &symbol;  
    visitDefault(symbol);  
    currentProcBlock = prevBlock;  
}

// ============================================================================
// 后处理和分析函数
// ============================================================================

bool DependencyVisitor::isControlVariable(const std::string& varName) {
    for (const auto& [otherName, otherInfo] : results) {
        for (const auto& assignment : otherInfo.assignments) {
            for (const auto& clause : assignment.path) {
                if (clause.expr.involvedSignals.count(varName)) {
                    return true;
                }
            }
        }
    }
    return false;
}

void DependencyVisitor::postProcess() {
    // Stage 1: 填充 fanOut 集合
    for (auto& [lhsName, info] : results) {
        for (const auto& assignment : info.assignments) {
            for (const auto& rhsName : assignment.drivingSignals) {
                if (results.count(rhsName)) {
                    results[rhsName].fanOut.insert(lhsName);
                }
            }
        }
    }

    // Stage 2: 清理赋值信息并计算聚合信息
    for (auto& [varName, info] : results) {
        std::set<AssignmentInfo> cleanedAssignments;
        
        for (const auto& assignment : info.assignments) {
            if (assignment.drivingSignals.empty() && 
                assignment.file.empty() && 
                assignment.line == 0) {
                continue;
            }
            if (assignment.type == "direct" && assignment.drivingSignals.empty()) {
                continue;
            }
            cleanedAssignments.insert(assignment);
        }
        
        info.assignments = cleanedAssignments;
        info.assignmentCount = info.assignments.size();
        
        // 计算 drivesOutput
        info.drivesOutput = false;
        for (const auto& fanOutName : info.fanOut) {
            if (results.count(fanOutName) && 
                results[fanOutName].direction == "output") {
                info.drivesOutput = true;
                break;
            }
        }
        
        // 计算 isControlVariable
        info.isControlVariable = isControlVariable(varName);
    }
}