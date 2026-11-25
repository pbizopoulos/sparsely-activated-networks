{ inputs, pkgs, ... }:
inputs.afairesi.lib.mkHtmlPackage {
  inherit pkgs;
  meta.description = "Interactive demonstration of sparsely activated networks and signal reconstruction";
  src = ./.;
}
