import * as vscode from 'vscode';

const DEFAULT_MODULES = [
    'tygent.integrations.google_ai',
    'tygent.integrations.anthropic',
    'tygent.integrations.huggingface',
    'tygent.integrations.microsoft_ai',
    'tygent.integrations.salesforce',
    'tygent.integrations.claude_code',
    'tygent.integrations.claude_code_cli',
    'tygent.integrations.gemini_cli',
    'tygent.integrations.openai_codex',
];

function buildInsertion(includeImport: boolean): string {
    const importBlock = includeImport ? 'import tygent\n\n' : '';
    const installLines = [
        'tygent.install([',
        ...DEFAULT_MODULES.map(module => `    "${module}",`),
        '])',
        '',
    ];
    return `${importBlock}${installLines.join('\n')}`;
}

export function activate(context: vscode.ExtensionContext) {
    const disposable = vscode.commands.registerCommand('tygent.enableAgentCursor', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('Open a Python file in Cursor to enable Tygent.');
            return;
        }

        const document = editor.document;
        if (document.languageId && document.languageId !== 'python') {
            vscode.window.showWarningMessage('The active file is not a Python file.');
            return;
        }

        const text = document.getText();
        if (text.includes('tygent.install(')) {
            vscode.window.showInformationMessage('Tygent is already installed in this file.');
            return;
        }

        const needsImport = !/^\s*import\s+tygent/m.test(text);
        let insertPosition = new vscode.Position(0, 0);
        let prefix = '';

        if (document.lineCount > 0) {
            const firstLine = document.lineAt(0);
            if (firstLine.text.startsWith('#!')) {
                insertPosition = firstLine.range.end;
                prefix = '\n';
            }
        }

        const insertion = prefix + buildInsertion(needsImport);
        const editApplied = await editor.edit(editBuilder => {
            editBuilder.insert(insertPosition, insertion);
        });

        if (editApplied) {
            vscode.window.showInformationMessage('Tygent conversion snippet inserted for Cursor.');
        } else {
            vscode.window.showErrorMessage('Unable to modify document.');
        }
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}
