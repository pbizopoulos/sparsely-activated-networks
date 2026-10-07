{ inputs, pkgs, ... }:
let
  python = pkgs.python3;
in
(inputs.perigrafo or inputs.self).lib.mkPythonPackage {
  inherit pkgs;
  executable = true;
  meta.description = "A Python package.";
  nativeBuildInputs = [ pkgs.texliveFull ];
  propagatedBuildInputs = [
    inputs.self.packages.${pkgs.stdenv.system}.sans
    inputs.self.packages.${pkgs.stdenv.system}.wfdb
    python.pkgs.pandas
    python.pkgs.torch
    python.pkgs.torchvision
  ];
  src = ./.;
  version = "0.0.0";
}
