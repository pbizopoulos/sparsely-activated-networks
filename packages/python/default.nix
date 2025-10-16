{
  inputs,
  pkgs ? import <nixpkgs> { },
}:
let
  wfdb = pkgs.python312Packages.buildPythonPackage rec {
    format = "wheel";
    pname = "wfdb";
    propagatedBuildInputs = [
      pkgs.python312Packages.fsspec
      pkgs.python312Packages.matplotlib
      pkgs.python312Packages.pandas
      pkgs.python312Packages.scipy
    ];
    pythonImportsCheck = [ pname ];
    src = pkgs.python312Packages.fetchPypi rec {
      inherit pname version format;
      dist = python;
      python = "py3";
      sha256 = "u9nSkbwgOLBYZhb82Acs/ckGrHDBhYsVEzeMQSmgEQ8=";
    };
    version = "4.3.0";
  };
in
pkgs.python312Packages.buildPythonPackage rec {
  installPhase = ''
    mkdir -p $out/bin
    cp ./main.py $out/bin/${pname}
    cp -r ./prm/ $out/bin/
  '';
  meta.mainProgram = pname;
  pname = builtins.baseNameOf src;
  propagatedBuildInputs = [
    inputs.self.packages.${pkgs.stdenv.system}.sans
    pkgs.python312Packages.torchvision-bin
    wfdb
  ];
  pyproject = false;
  src = ./.;
  version = "0.0.0";
}
