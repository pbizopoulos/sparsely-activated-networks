{ inputs, pkgs, ... }:
let
  nativeDeps = [ ];
  pname = baseNameOf ./.;
  python = pkgs.python3;
  pythonDeps = [
    inputs.self.packages.${pkgs.stdenv.system}.sans
    inputs.self.packages.${pkgs.stdenv.system}.wfdb
    python.pkgs.pandas
    python.pkgs.torch
    python.pkgs.torchvision
  ];
in
python.pkgs.buildPythonPackage {
  inherit pname;
  installPhase = ''
    install -Dm644 main.py "$out/${python.sitePackages}/$pname.py"
    install -Dm755 main.py "$out/bin/$pname"
    if [ -d prm ]; then
      cp -R prm/ "$out/${python.sitePackages}/"
      cp -R prm/ "$out/bin/"
    fi
  '';
  meta.mainProgram = pname;
  nativeBuildInputs = nativeDeps;
  passthru.python = python;
  propagatedBuildInputs = pythonDeps;
  pyproject = false;
  src = ./.;
  strictDeps = true;
  version = "0.0.0";
}
