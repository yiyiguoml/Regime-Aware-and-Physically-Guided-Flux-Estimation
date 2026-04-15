function [m_der, m_mo] = py_init_mo(varargin)
% PY_INIT_MO  Initialize Python (OutOfProcess) and import local modules.
%
% Usage:
%   [m_der, m_mo] = py_init_mo();
%   [m_der, m_mo] = py_init_mo("PythonExe", "/path/to/python3");
%
% Repo layout assumed:
%   <repo_root>/utils/py_init_mo.m
%   <repo_root>/physics/derived.py
%   <repo_root>/physics/mo.py

    p = inputParser;
    addParameter(p, "PythonExe", "", @(x)isstring(x)||ischar(x));
    parse(p, varargin{:});
    pyexe = string(p.Results.PythonExe);

    % --- (1) Locate repo root relative to this file (no hard-coded absolute path)
    thisFile = string(mfilename("fullpath"));            % .../<repo>/utils/py_init_mo.m
    utilsDir = string(fileparts(thisFile));              % .../<repo>/utils
    repoRoot = string(fileparts(utilsDir));              % .../<repo>
    moDir    = fullfile(repoRoot, "physics");

    if ~isfolder(moDir)
        error("py_init_mo:PathNotFound", "Cannot find python_module folder at: %s", moDir);
    end

    % --- (2) Start / bind Python (clean state)
    try terminate(pyenv); catch, end

    if strlength(pyexe) > 0
        pyenv("Version", pyexe, "ExecutionMode", "OutOfProcess");
    else
        pyenv("ExecutionMode", "OutOfProcess");
    end

    % --- (3) Make sys.path deterministic (avoid '' current-dir surprises)
    try py.sys.path.remove(""); catch, end
    try py.sys.path.remove(''); catch, end

    moDirPy = string(moDir);
    if ~any(string(py.sys.path) == moDirPy)
        insert(py.sys.path, int32(0), moDirPy);
    end

    py.importlib.invalidate_caches();

    % --- (4) Import
    m_der = py.importlib.import_module("derived");
    m_mo  = py.importlib.import_module("mo");

    % --- (5) Sanity prints (useful for reproducibility logs)
    try
        fprintf('[py_init] Python: %s\n', char(pyenv().Executable));
        fprintf('[py_init] derived: %s\n', char(py.getattr(m_der, "__file__")));
        fprintf('[py_init] mo     : %s\n', char(py.getattr(m_mo,  "__file__")));
    catch
    end
end