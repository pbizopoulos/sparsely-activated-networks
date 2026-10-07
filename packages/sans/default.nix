{ inputs, pkgs, ... }:
let
  python = pkgs.python3;
in
(inputs.afairesi or inputs.self).lib.mkPythonPackage {
  inherit pkgs;
  executable = false;
  meta.description = "A Python package.";
  propagatedBuildInputs = [ python.pkgs.torch ];
  src = ./.;
  version = "0.0.0";
}
