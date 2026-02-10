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
    const disposable = vscode.commands.registerCommand('tygent.enableAgent', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        const doc = editor.document;
        const text = doc.getText();
        if (text.includes('tygent.install(')) {
            vscode.window.showInformationMessage('Tygent already enabled in this file.');
            return;
        }
        editor.edit(editBuilder => {
            const firstLine = doc.lineAt(0);
            let position = new vscode.Position(0, 0);
            let prefix = '';
            if (firstLine.text.startsWith('#!')) {
                position = firstLine.range.end;
                prefix = '\n';
            }
            const needsImport = !/^\s*import\s+tygent/m.test(text);
            editBuilder.insert(position, `${prefix}${buildInsertion(needsImport)}`);
        });
    });
    context.subscriptions.push(disposable);
}

export function deactivate() {}
