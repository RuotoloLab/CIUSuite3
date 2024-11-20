; Script generated by the Inno Script Studio Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "CIUSuite 3"
#define MyAppDirName "CIUSuite3"
#define MyAppVersion "1.0"
#define MyAppPublisher "University of Michigan"
#define MyAppURL "https://github.com/RuotoloLab/CIUSuite2/releases/tag/v3.0.0"
#define MyAppExeName "CIU3_Main.exe"
#define ReadmeName "CIUSuite3_Manual.pdf"
#define IconName "CIUSuite3_win10.ico"
#define AgilentExtName "MIDAC_CIU_Extractor"
#define TWIMExName "TWIMExtract"
#define SourceDir "C:\Users\caror\CIUSuite3"
#define OutputDir "C:\Users\caror\CIUSuite3\Output"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{7A672883-4651-4D6B-96AA-E530E7140B34}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={pf}\{#MyAppDirName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile={#SourceDir}\LICENSE.txt
;InfoAfterFile=C:\Users\dpolasky\PycharmProjects\CIUSuite2\README.txt
OutputDir={#OutputDir}
OutputBaseFilename=CIUSuite3_Setup
Compression=lzma
SolidCompression=yes
UsePreviousAppDir=False

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Final version includes both TWIMExtract and MIDAC Extractor
Source: "{#SourceDir}\dist\CIU3_Main\CIU3_Main.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\dist\CIU3_Main\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\{#ReadmeName}"; DestDir: "{app}"; Flags: isreadme ignoreversionSource: "{#SourceDir}\classification_template_example.csv"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\CIUSuite3_win10.ico"; DestDir: "{app}"; DestName: "{#IconName}"; Flags: ignoreversion
Source: "{#SourceDir}\README.txt"; DestDir: "{app}"; Flags: ignoreversion 
; Source: "C:\Users\dpolasky\Desktop\Data Tools and Src Code\_Agilent CIU Extractor\_SIMPLE_versionForDistribution\release\*"; DestDir: "{app}\Agilent_Extractor"; Flags: ignoreversion
Source: "{#SourceDir}\Agilent_Extractor\*"; DestDir: "{app}\_internal\Agilent_Extractor"; Flags: ignoreversion
Source: "{#SourceDir}\Breuker_Extractor\*"; DestDir: "{app}\_internal\Breuker_Extractor"; Flags: ignoreversion
; Source: "C:\Users\dpolasky\PycharmProjects\CIUSuite2\TWIMExtract\*"; DestDir: "{app}\TWIMExtract"; Flags: ignoreversion recursesubdirs
Source: "{#SourceDir}\TWIMExtract\*"; DestDir: "{app}\_internal\TWIMExtract"; Permissions: users-modify; Flags: ignoreversion recursesubdirs
Source: "{#SourceDir}\CIU3_param_info.csv"; DestDir: "{commonappdata}\{#MyAppDirName}"; Permissions: users-modify; Flags: ignoreversion
Source: "{#SourceDir}\FromRaw_to_CCS-CIU_template.csv"; DestDir: "{commonappdata}\{#MyAppDirName}"; Permissions: users-modify; Flags: ignoreversion
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "cmd"; Parameters: "/k ""{app}\{#MyAppExeName}"""; IconFilename: "{app}\{#IconName}"
Name: "{group}\{#ReadmeName}"; Filename: "{app}\{#ReadmeName}"
Name: "{group}\{#AgilentExtName}"; Filename: "{app}\Agilent_Extractor\{#AgilentExtName}.exe"
Name: "{group}\{#TWIMExName}"; Filename: "{app}\TWIMExtract\runTWIMExtract.bat"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\{#IconName}"
; Name: "{commondesktop}\{#ReadmeName}"; Filename: "{app}\{#ReadmeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
