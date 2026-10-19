{ inputs, pkgs, ... }:
let
  python = pkgs.python3;
in
(inputs.afairesi or inputs.self).lib.mkPythonPackage {
  inherit pkgs;
  executable = true;
  meta = {
    description = "Sparsely activated networks";
    homepage = "https://arxiv.org/abs/1907.06592";
    longDescription = ''
      Publication: https://ieeexplore.ieee.org/abstract/document/9072295/
      Project website: https://pbizopoulos.github.io/sparsely-activated-networks/
    '';
  };
  nativeBuildInputs = [ pkgs.texliveFull ];
  propagatedBuildInputs = [
    inputs.self.packages.${pkgs.stdenv.system}.wfdb
    python.pkgs.pandas
    python.pkgs.torch
    python.pkgs.torchvision
  ];
  src = ./.;
}
