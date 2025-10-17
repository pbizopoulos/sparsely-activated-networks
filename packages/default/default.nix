{
  inputs,
  pkgs ? import <nixpkgs> { },
}:
let
  pythonEnv = pkgs.python312.withPackages (_ps: [
    inputs.self.packages.${pkgs.stdenv.system}.sans
    pkgs.python312Packages.torchvision-bin
    wfdb
  ]);
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
pkgs.stdenv.mkDerivation rec {
  buildInputs = [
    pkgs.texlive.combined.scheme-full
    pythonEnv
  ];
  installPhase = ''
    mkdir -p $out/bin
    echo '#!/usr/bin/env bash
      set -e
      package_dir=$HOME/github.com/pbizopoulos/sparsely-activated-networks/packages/default
      tmp_dir=$(mktemp -d)
      cp -r ${src}/* "$tmp_dir"
      cd "$tmp_dir"
      ${pythonEnv}/bin/python ./main.py
      ${pkgs.texlive.combined.scheme-full}/bin/latexmk -outdir=$package_dir/tmp -pdf ./ms.tex
      ' > $out/bin/${pname}
    chmod +x $out/bin/${pname}
  '';
  meta.mainProgram = pname;
  pname = builtins.baseNameOf src;
  src = ./.;
  version = "0.0.0";
}
