import jax

# Conservation-law checks need double precision to be meaningful.
jax.config.update("jax_enable_x64", True)
