const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});

    // Standard optimization options
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    // const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSafe });

    // Add the executable for the C project
    const exe = b.addExecutable(.{
        .name = "nn-project",
        .target = target,
        .optimize = optimize,
    });

    // Add additional C source files to the executable
    exe.addCSourceFile(.{ .file = b.path("src/main.c") });
    exe.addCSourceFile(.{ .file = b.path("src/nn.c") });
    exe.addCSourceFile(.{ .file = b.path("src/mnist_loader.c") });

    // Add include directories
    exe.addIncludePath(b.path("include"));

    // Link system libraries if required
    // Uncomment if your project depends on standard libraries like math
    exe.linkSystemLibrary("c");

    // Install the executable as an artifact
    b.installArtifact(exe);

    // Create a Run step in the build graph
    const run_cmd = b.addRunArtifact(exe);

    // Make the run step depend on the install step
    run_cmd.step.dependOn(b.getInstallStep());

    // Allow passing arguments to the application
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Add a run step for executing the program
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Unit testing steps (if applicable)
    // You can extend this to add unit tests for your C project, if needed.
    // Example placeholder for unit testing:
    // const test_step = b.step("test", "Run unit tests");
    // test_step.dependOn(&run_cmd.step);
}
