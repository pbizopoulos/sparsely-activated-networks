{
  pkgs ? import <nixpkgs> { },
}:
let
  python = pkgs.python3;
  pythonDeps = [
    python.pkgs.aiohttp
    python.pkgs.fsspec
    python.pkgs.matplotlib
    python.pkgs.pandas
    python.pkgs.requests
    python.pkgs.scipy
    python.pkgs.soundfile
  ];
in
python.pkgs.buildPythonPackage rec {
  format = "wheel";
  pname = "wfdb";
  propagatedBuildInputs = pythonDeps;
  pythonImportsCheck = [ pname ];
  src = python.pkgs.fetchPypi rec {
    inherit pname version format;
    dist = python;
    python = "py3";
    sha256 = "u9nSkbwgOLBYZhb82Acs/ckGrHDBhYsVEzeMQSmgEQ8=";
  };
  version = "4.3.0";
}
