{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        check-python-script = pkgs.python3Packages.buildPythonPackage rec {
          name = "check-python-script";
          version = "0.0";
          format = "pyproject";
          src = fetchTarball rec {
            url = "https://api.github.com/repos/pbizopoulos/check-python-script/tarball/main#subdirectory=python";
            sha256 = "1rpzxyrkp8y86m7zryrvbhzwzifrij6y18gyhlkpgav7lp7rwxrk";
          };
          preBuild = ''
            cd python/
          '';
          propagatedBuildInputs = [
            pkgs.python3Packages.fire
            pkgs.python3Packages.libcst
            pkgs.python3Packages.setuptools
          ];
        };
        wfdb = pkgs.python3Packages.buildPythonPackage rec {
          pname = "wfdb";
          version = "4.1.2";
          format = "wheel";
          src = pkgs.python3Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "V+9kMJ7HeTuxFhHGRtEp8LGTa1mJR/7YwU+Qo1ZlBT0=";
            dist = python;
            python = "py3";
          };
          propagatedBuildInputs = [
            pkgs.python3Packages.matplotlib
            pkgs.python3Packages.pandas
            pkgs.python3Packages.scipy
          ];
        };
        packages = with pkgs; [
          python3Packages.torch-bin
          python3Packages.torchvision-bin
          python3Packages.types-requests
          wfdb
        ];
      in
      with pkgs;
      {
        devShells.default = pkgs.mkShell { buildInputs = packages; };
        devShells.check = pkgs.mkShell {
          buildInputs = packages ++ [
            check-python-script
            djlint
            mypy
            nixfmt-rfc-style
            ruff
          ];
        };
      }
    );
}
