{
  inputs.perigrafo.url = "github:perigrafo/perigrafo";
  outputs =
    inputs:
    inputs.perigrafo.blueprint {
      inherit inputs;
      nixpkgs.config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    }
    // {
      inherit (inputs.perigrafo) formatter;
    };
}
