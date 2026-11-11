{ inputs, pkgs, ... }:
inputs.afairesi.lib.mkHtmlPackage {
  inherit pkgs;
  meta.description = "An HTML, CSS, and JavaScript template package.";
  src = ./.;
}
